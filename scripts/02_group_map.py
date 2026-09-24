#!/usr/bin/env python3
"""
SafeGI-VLM  |  Phase 0, step 2  |  scripts/02_group_map.py

Turns near-duplicate PAIRS into GROUPS, and tells you how much independent
data you really have.

Why this exists
---------------
HyperKvasir's frames were extracted from colonoscopy videos, so the archive
contains many near-identical frames of the same lesion under unrelated
filenames. If two frames of one polyp land in different splits, that polyp
is in training and in test, and the generalisation claim is void. Grouping
has to happen before splitting, and it has to be transitive: if A~B and
B~C, then A, B and C are one group even when A and C look different.

Two safeguards, because getting this wrong in either direction is costly.

  Dual hash. A pair is only a duplicate if pHash AND dHash both agree.
  Endoscopic frames are globally similar - pink mucosa, dark lumen, circular
  vignette - so a single perceptual hash produces false positives on frames
  that merely look alike. Requiring two independent hashes to agree cuts
  that sharply. Use --single-hash to see the difference.

  Chaining check. Transitive closure at a loose threshold can collapse the
  whole dataset into one blob through chains of marginal similarities. The
  threshold sweep reports the largest component at each threshold; if it
  runs away, the threshold is too loose and the sweep shows exactly where.

Usage
-----
    python scripts/02_group_map.py --manifests data/manifests --hamming 4
    python scripts/02_group_map.py --manifests data/manifests --sweep
    python scripts/02_group_map.py --manifests data/manifests --hamming 4 \
        --sample-pairs 40 --root "E:/OurRe"     # contact sheet to eyeball
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SWEEP_THRESHOLDS = (0, 2, 4, 6, 8, 10, 12)


# --- union-find --------------------------------------------------------------

class Union:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# --- hamming -----------------------------------------------------------------

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _codes(series: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in series], dtype=np.uint64)


def _block_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=2, dtype=np.uint8)


def find_pairs(ph: np.ndarray, dh: np.ndarray | None, threshold: int,
               block: int = 512) -> list[tuple[int, int]]:
    """Index pairs within `threshold` on pHash, and on dHash too when given."""
    n = len(ph)
    pairs: list[tuple[int, int]] = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        ok = _block_hamming(ph[start:stop], ph) <= threshold
        if dh is not None:
            ok &= _block_hamming(dh[start:stop], dh) <= threshold
        ii, jj = np.nonzero(ok)
        for i, j in zip(ii, jj):
            gi, gj = start + int(i), int(j)
            if gi < gj:
                pairs.append((gi, gj))
    return pairs


# --- grouping ----------------------------------------------------------------

def build_groups(df: pd.DataFrame, pairs: list[tuple[int, int]],
                 sha_col: str = "sha256") -> np.ndarray:
    """Connected components over near-duplicate pairs plus exact duplicates."""
    uf = Union(len(df))

    by_sha: dict[str, int] = {}
    for idx, sha in enumerate(df[sha_col]):
        if sha in by_sha:
            uf.join(by_sha[sha], idx)
        else:
            by_sha[sha] = idx

    for a, b in pairs:
        uf.join(a, b)

    roots = np.array([uf.find(i) for i in range(len(df))])
    _, group_id = np.unique(roots, return_inverse=True)
    return group_id


def describe(df: pd.DataFrame, group_id: np.ndarray) -> dict:
    sizes = Counter(group_id.tolist())
    hist = Counter(sizes.values())
    multi = {g: s for g, s in sizes.items() if s > 1}

    biggest = sorted(sizes.items(), key=lambda kv: -kv[1])[:10]
    big_detail = []
    for g, s in biggest:
        if s < 2:
            continue
        member_paths = df.rel_path[group_id == g].tolist()
        labels = Counter(df.class_label[group_id == g].tolist())
        big_detail.append({
            "group": int(g), "size": int(s),
            "classes": {str(k): int(v) for k, v in labels.items()},
            "examples": member_paths[:4],
        })

    per_class_groups = defaultdict(set)
    per_class_images = Counter()
    for lbl, g in zip(df.class_label, group_id):
        per_class_groups[str(lbl)].add(int(g))
        per_class_images[str(lbl)] += 1

    return {
        "n_images": int(len(df)),
        "n_groups": int(len(sizes)),
        "n_singleton_groups": int(hist.get(1, 0)),
        "n_multi_groups": int(len(multi)),
        "n_images_in_multi_groups": int(sum(multi.values())),
        "largest_group": int(max(sizes.values())),
        "size_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "largest_groups": big_detail,
        "per_class": {
            c: {"images": int(per_class_images[c]), "groups": int(len(per_class_groups[c]))}
            for c in sorted(per_class_images)
        },
    }


# --- optional visual check ----------------------------------------------------

def sample_sheet(df: pd.DataFrame, pairs: list[tuple[int, int]], root: Path,
                 out: Path, n: int, seed: int = 0) -> Path | None:
    """Contact sheet of sampled flagged pairs, so a human can sanity-check."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    if not pairs:
        return None

    rng = np.random.default_rng(seed)
    picked = [pairs[i] for i in rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)]

    thumb, pad, lab = 220, 6, 16
    rows = len(picked)
    sheet = Image.new("RGB", (2 * thumb + 3 * pad, rows * (thumb + lab) + (rows + 1) * pad),
                      (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    for r, (a, b) in enumerate(picked):
        y = pad + r * (thumb + lab + pad)
        for c, idx in enumerate((a, b)):
            p = root / df.rel_path.iloc[idx]
            x = pad + c * (thumb + pad)
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb, thumb), Image.LANCZOS)
                sheet.paste(im, (x, y))
            except Exception:
                d.rectangle([x, y, x + thumb, y + thumb], fill=(220, 220, 220))
            d.text((x + 2, y + thumb + 2),
                   f"{df.class_label.iloc[idx]} | {Path(df.rel_path.iloc[idx]).name[:26]}",
                   fill=(30, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


# --- main ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Near-duplicate grouping for split assignment")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--hamming", type=int, default=4)
    ap.add_argument("--single-hash", action="store_true", help="pHash only (not recommended)")
    ap.add_argument("--sweep", action="store_true", help="report the threshold sweep and stop")
    ap.add_argument("--sample-pairs", type=int, default=0, help="write a contact sheet of N pairs")
    ap.add_argument("--root", default="", help="data root, needed for --sample-pairs")
    args = ap.parse_args()

    man = Path(args.manifests).expanduser().resolve()
    src = man / "inventory.parquet"
    if not src.exists():
        src = man / "inventory.csv"
    if not src.exists():
        sys.exit(f"No inventory found in {man}. Run scripts/00_inventory.py first.")

    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)

    # Masks carry no perceptual hash and follow their image's group later.
    if "phash" not in df.columns:
        sys.exit("Inventory has no phash column. Re-run 00_inventory.py without --no-hash.")
    df = df[df.phash.fillna("") != ""].reset_index(drop=True)
    df["class_label"] = df.class_label.fillna("(none)")
    print(f"{len(df):,} hashed images from {src.name}")

    ph = _codes(df.phash)
    dh = None if args.single_hash else _codes(df.dhash)
    if dh is not None:
        print("using pHash AND dHash agreement")

    if args.sweep:
        print(f"\n{'H':>3} {'pairs':>9} {'groups':>8} {'multi':>7} {'in multi':>9} {'largest':>8}")
        rows = []
        for t in SWEEP_THRESHOLDS:
            pairs = find_pairs(ph, dh, t)
            gid = build_groups(df, pairs)
            s = describe(df, gid)
            rows.append({"hamming": t, "pairs": len(pairs), **{
                k: s[k] for k in ("n_groups", "n_multi_groups",
                                  "n_images_in_multi_groups", "largest_group")}})
            print(f"{t:>3} {len(pairs):>9,} {s['n_groups']:>8,} {s['n_multi_groups']:>7,} "
                  f"{s['n_images_in_multi_groups']:>9,} {s['largest_group']:>8,}")
        (man / "grouping_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwritten: {man / 'grouping_sweep.json'}")
        print("\nPick the largest threshold at which 'largest' is still plausible as one")
        print("lesion or one short video clip. A sudden jump means chaining has begun.")
        return

    pairs = find_pairs(ph, dh, args.hamming)
    gid = build_groups(df, pairs)
    summary = describe(df, gid)
    summary.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hamming_threshold": args.hamming,
        "dual_hash": not args.single_hash,
        "n_pairs": len(pairs),
    })

    out = df[["rel_path", "dataset", "class_label", "sha256"]].copy()
    out["group_id"] = gid
    try:
        out.to_parquet(man / "group_map.parquet", index=False)
        written = man / "group_map.parquet"
    except Exception:
        written = man / "group_map.csv"
        out.to_csv(written, index=False)

    (man / "grouping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"grouping at H<={args.hamming}"
          f"{' (dual hash)' if not args.single_hash else ' (pHash only)'}")
    print("=" * 66)
    print(f"images              : {summary['n_images']:,}")
    print(f"pairs               : {summary['n_pairs']:,}")
    print(f"independent groups  : {summary['n_groups']:,}   "
          f"<- this is your real sample size")
    print(f"  singletons        : {summary['n_singleton_groups']:,}")
    print(f"  multi-image       : {summary['n_multi_groups']:,} groups "
          f"covering {summary['n_images_in_multi_groups']:,} images")
    print(f"largest group       : {summary['largest_group']:,} images")

    print("\ngroup size histogram (size: count)")
    for k, v in list(summary["size_histogram"].items())[:12]:
        print(f"  {k:>4}: {v:,}")

    print("\nper class: images -> independent groups")
    for c, s in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]["images"]):
        loss = 100 * (1 - s["groups"] / s["images"]) if s["images"] else 0
        print(f"  {c:<32} {s['images']:>6,} -> {s['groups']:>6,}  ({loss:4.1f}% collapsed)")

    if summary["largest_groups"]:
        print("\nlargest groups:")
        for g in summary["largest_groups"][:5]:
            print(f"  size {g['size']:>3}  classes={g['classes']}")
            print(f"           {Path(g['examples'][0]).name}")

    if args.sample_pairs and args.root:
        p = sample_sheet(df, pairs, Path(args.root), man.parent / "interim" /
                         "group_check" / f"pairs_H{args.hamming}.jpg", args.sample_pairs)
        if p:
            print(f"\nvisual check: {p}")
            print("Open it. Each row is one flagged pair. If rows show clearly")
            print("different scenes, the threshold is too loose.")

    print(f"\nwritten: {written}")
    print(f"written: {man / 'grouping_summary.json'}")


if __name__ == "__main__":
    main()

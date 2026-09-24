#!/usr/bin/env python3
"""
SafeGI-VLM  |  Phase 0, step 1  |  scripts/00_inventory.py

Inventories the source-domain datasets on disk and writes a manifest.

This answers, with evidence rather than assumption:
  - what is actually in E:\\OurRe, and under what folder structure
  - how many images per dataset and per HyperKvasir class
  - image resolutions, formats, colour modes, and corrupt/unreadable files
  - Kvasir-SEG image/mask pairing integrity, and mask area statistics
  - exact-duplicate and near-duplicate pairs, within and across datasets

It does NOT assign splits. Splits come later, after Kvasir-VQA and the
external sets are present, because duplicates must be grouped across all
of them before anything is assigned.

Usage
-----
    python scripts/00_inventory.py --root "E:/OurRe" --out data/manifests

    # faster first pass, skips perceptual hashing:
    python scripts/00_inventory.py --root "E:/OurRe" --out data/manifests --no-hash

Requires: pillow numpy pandas tqdm imagehash
    pip install pillow numpy pandas tqdm imagehash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # endoscopy frames are small; silence the bomb warning

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Folder-name fragments that identify a dataset. Matching is case-insensitive
# and checked against the whole relative path, so it survives the various
# ways these archives unpack.
DATASET_PATTERNS = [
    ("kvasir-seg-masks", ("kvasir-seg", "masks")),
    ("kvasir-seg-images", ("kvasir-seg", "images")),
    ("hyperkvasir-labeled", ("labeled-images",)),
    ("hyperkvasir-segmented", ("segmented-images",)),
]


# --------------------------------------------------------------------------
# classification of a file
# --------------------------------------------------------------------------

def classify(rel_path: Path) -> tuple[str, str | None]:
    """Return (dataset_tag, class_label) for a file, from its path."""
    parts_lower = [p.lower() for p in rel_path.parts]
    joined = "/".join(parts_lower)

    for tag, needles in DATASET_PATTERNS:
        if all(n in joined for n in needles):
            # HyperKvasir labeled-images nests as
            #   labeled-images/<lower-gi|upper-gi>-tract/<category>/<class>/x.jpg
            label = None
            if tag.startswith("hyperkvasir"):
                label = rel_path.parent.name
            return tag, label

    return "unclassified", rel_path.parent.name


# --------------------------------------------------------------------------
# per-file probing
# --------------------------------------------------------------------------

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def probe_image(path: Path) -> dict:
    """Open an image and report its properties, or the error if unreadable."""
    try:
        with Image.open(path) as im:
            im.verify()  # cheap integrity check; invalidates the handle
        with Image.open(path) as im:
            width, height = im.size
            mode, fmt = im.mode, im.format
        return {
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
            "readable": True,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - we want every failure recorded
        return {
            "width": -1,
            "height": -1,
            "mode": "",
            "format": "",
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def mask_stats(path: Path) -> dict:
    """Foreground fraction and bounding box for a binary segmentation mask."""
    try:
        with Image.open(path) as im:
            arr = np.array(im.convert("L"))
        fg = arr > 127
        n_fg = int(fg.sum())
        if n_fg == 0:
            return {"mask_area_frac": 0.0, "mask_empty": True,
                    "mask_bbox": "", "mask_centroid_quadrant": ""}
        ys, xs = np.nonzero(fg)
        h, w = arr.shape
        cy, cx = ys.mean() / h, xs.mean() / w
        quadrant = ("top" if cy < 0.5 else "bottom") + "-" + ("left" if cx < 0.5 else "right")
        return {
            "mask_area_frac": round(n_fg / arr.size, 6),
            "mask_empty": False,
            "mask_bbox": f"{xs.min()},{ys.min()},{xs.max()},{ys.max()}",
            "mask_centroid_quadrant": quadrant,
        }
    except Exception as exc:  # noqa: BLE001
        return {"mask_area_frac": -1.0, "mask_empty": False,
                "mask_bbox": "", "mask_centroid_quadrant": f"ERROR {exc}"}


# --------------------------------------------------------------------------
# main walk
# --------------------------------------------------------------------------

def walk(root: Path, do_hash: bool) -> pd.DataFrame:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        sys.exit(f"No image files found under {root}. Check the path and that extraction finished.")

    print(f"Found {len(files):,} image files under {root}")

    try:
        import imagehash
    except ImportError:
        if do_hash:
            sys.exit("imagehash not installed. Run: pip install imagehash   (or pass --no-hash)")
        imagehash = None

    rows = []
    for path in tqdm(files, desc="probing", unit="img"):
        rel = path.relative_to(root)
        dataset, label = classify(rel)
        rec = {
            "rel_path": str(rel).replace("\\", "/"),
            "dataset": dataset,
            "class_label": label,
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }
        rec.update(probe_image(path))

        if rec["readable"] and dataset == "kvasir-seg-masks":
            rec.update(mask_stats(path))

        if do_hash and rec["readable"] and imagehash is not None and "masks" not in dataset:
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    rec["phash"] = str(imagehash.phash(im))
                    rec["dhash"] = str(imagehash.dhash(im))
            except Exception as exc:  # noqa: BLE001
                rec["phash"] = rec["dhash"] = ""
                rec["error"] = f"hash failed: {exc}"

        rows.append(rec)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# analyses over the inventory
# --------------------------------------------------------------------------

def pairing_report(df: pd.DataFrame) -> dict:
    """Kvasir-SEG ships images/ and masks/ with matching stems. Verify that."""
    imgs = df[df.dataset == "kvasir-seg-images"]
    masks = df[df.dataset == "kvasir-seg-masks"]
    if imgs.empty and masks.empty:
        return {"checked": False, "reason": "Kvasir-SEG images/masks folders not identified"}

    img_stems = {Path(p).stem for p in imgs.rel_path}
    mask_stems = {Path(p).stem for p in masks.rel_path}

    size_mismatch = []
    by_stem_img = {Path(r.rel_path).stem: (r.width, r.height) for r in imgs.itertuples()}
    for r in masks.itertuples():
        stem = Path(r.rel_path).stem
        if stem in by_stem_img and by_stem_img[stem] != (r.width, r.height):
            size_mismatch.append(stem)

    empty_masks = masks[masks.get("mask_empty", False) == True]  # noqa: E712

    return {
        "checked": True,
        "n_images": len(imgs),
        "n_masks": len(masks),
        "images_without_mask": sorted(img_stems - mask_stems)[:50],
        "n_images_without_mask": len(img_stems - mask_stems),
        "masks_without_image": sorted(mask_stems - img_stems)[:50],
        "n_masks_without_image": len(mask_stems - img_stems),
        "n_size_mismatch": len(size_mismatch),
        "size_mismatch_examples": size_mismatch[:20],
        "n_empty_masks": int(len(empty_masks)),
        "empty_mask_examples": empty_masks.rel_path.head(20).tolist(),
    }


def duplicate_report(df: pd.DataFrame, hamming_threshold: int = 4) -> dict:
    """Exact duplicates by sha256, near-duplicates by perceptual hash."""
    out: dict = {}

    exact = defaultdict(list)
    for r in df.itertuples():
        exact[r.sha256].append(r.rel_path)
    exact_groups = [v for v in exact.values() if len(v) > 1]
    out["n_exact_duplicate_groups"] = len(exact_groups)
    out["exact_duplicate_examples"] = exact_groups[:25]

    if "phash" not in df.columns:
        out["near_duplicates"] = "skipped (--no-hash)"
        return out

    sub = df[(df.phash.fillna("") != "")].reset_index(drop=True)
    if sub.empty:
        out["near_duplicates"] = "no hashes computed"
        return out

    # 64-bit hex hashes -> bit matrix, so Hamming distance is a matrix op.
    unpacked = np.unpackbits(
        np.array([[int(h[i:i + 2], 16) for i in range(0, 16, 2)] for h in sub.phash],
                 dtype=np.uint8),
        axis=1,
    )

    pairs = []
    n = len(unpacked)
    block = 2048
    for start in tqdm(range(0, n, block), desc="near-dup", unit="blk"):
        stop = min(start + block, n)
        d = (unpacked[start:stop, None, :] != unpacked[None, :, :]).sum(axis=2)
        idx_i, idx_j = np.nonzero(d <= hamming_threshold)
        for i, j in zip(idx_i, idx_j):
            gi = start + int(i)
            gj = int(j)
            if gi < gj:
                pairs.append((gi, gj, int(d[i, j])))

    cross, within = [], []
    for gi, gj, dist in pairs:
        a, b = sub.iloc[gi], sub.iloc[gj]
        entry = {"a": a.rel_path, "b": b.rel_path,
                 "a_dataset": a.dataset, "b_dataset": b.dataset, "hamming": dist}
        (cross if a.dataset != b.dataset else within).append(entry)

    out["hamming_threshold"] = hamming_threshold
    out["n_near_duplicate_pairs"] = len(pairs)
    out["n_within_dataset_pairs"] = len(within)
    out["n_cross_dataset_pairs"] = len(cross)
    out["cross_dataset_examples"] = cross[:50]
    out["within_dataset_examples"] = within[:25]
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SafeGI-VLM Phase 0 dataset inventory")
    ap.add_argument("--root", required=True, help='Data root, e.g. "E:/OurRe"')
    ap.add_argument("--out", default="data/manifests", help="Output directory")
    ap.add_argument("--no-hash", action="store_true", help="Skip perceptual hashing")
    ap.add_argument("--hamming", type=int, default=4, help="Near-duplicate threshold (default 4)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = walk(root, do_hash=not args.no_hash)

    inv_path = outdir / "inventory.parquet"
    try:
        df.to_parquet(inv_path, index=False)
    except Exception:  # pyarrow missing
        inv_path = outdir / "inventory.csv"
        df.to_csv(inv_path, index=False)

    readable = df[df.readable]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "n_files": int(len(df)),
        "n_readable": int(len(readable)),
        "n_unreadable": int((~df.readable).sum()),
        "unreadable_examples": df[~df.readable][["rel_path", "error"]].head(20).to_dict("records"),
        "by_dataset": {k: int(v) for k, v in Counter(df.dataset).items()},
        "by_dataset_and_class": {
            f"{d}/{c}": int(n)
            for (d, c), n in Counter(zip(df.dataset, df.class_label)).items()
        },
        "formats": {k: int(v) for k, v in Counter(readable.format).items()},
        "modes": {k: int(v) for k, v in Counter(readable["mode"]).items()},
        "resolution": {
            "width_min": int(readable.width.min()), "width_max": int(readable.width.max()),
            "height_min": int(readable.height.min()), "height_max": int(readable.height.max()),
            "distinct_resolutions": int(len(set(zip(readable.width, readable.height)))),
            "most_common": [
                {"wxh": f"{w}x{h}", "n": int(n)}
                for (w, h), n in Counter(zip(readable.width, readable.height)).most_common(10)
            ],
        },
        "total_gb": round(float(df.bytes.sum()) / 1e9, 3),
        "kvasir_seg_pairing": pairing_report(df),
        "duplicates": duplicate_report(df, args.hamming),
    }

    masks = df[df.dataset == "kvasir-seg-masks"]
    if "mask_area_frac" in masks.columns and not masks.empty:
        areas = masks.mask_area_frac[masks.mask_area_frac >= 0]
        summary["mask_area_fraction"] = {
            "n": int(len(areas)),
            "min": round(float(areas.min()), 6),
            "p25": round(float(areas.quantile(0.25)), 6),
            "median": round(float(areas.median()), 6),
            "p75": round(float(areas.quantile(0.75)), 6),
            "max": round(float(areas.max()), 6),
            "n_over_40pct": int((areas > 0.40).sum()),
            "note": "Polyps above ~0.40 area fraction cannot get a size-matched "
                    "non-overlapping control mask and are excluded from Phase 6.",
        }
        summary["mask_centroid_quadrants"] = {
            k: int(v) for k, v in Counter(masks.mask_centroid_quadrant.dropna()).items()
        }

    sum_path = outdir / "inventory_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- console report -------------------------------------------------
    print("\n" + "=" * 68)
    print("SafeGI-VLM  |  Phase 0 inventory")
    print("=" * 68)
    print(f"root            : {root}")
    print(f"files           : {summary['n_files']:,}  "
          f"({summary['n_unreadable']} unreadable, {summary['total_gb']} GB)")
    print("\nby dataset:")
    for k, v in sorted(summary["by_dataset"].items()):
        print(f"  {k:<28} {v:>7,}")

    hk = {k: v for k, v in summary["by_dataset_and_class"].items() if k.startswith("hyperkvasir")}
    if hk:
        print("\nHyperKvasir classes (candidate polyp-free frames in bold in your head):")
        for k, v in sorted(hk.items(), key=lambda x: -x[1]):
            print(f"  {k.split('/', 1)[1]:<34} {v:>7,}")

    pr = summary["kvasir_seg_pairing"]
    if pr.get("checked"):
        print(f"\nKvasir-SEG pairing: {pr['n_images']} images / {pr['n_masks']} masks")
        print(f"  images without mask : {pr['n_images_without_mask']}")
        print(f"  masks without image : {pr['n_masks_without_image']}")
        print(f"  size mismatches     : {pr['n_size_mismatch']}")
        print(f"  empty masks         : {pr['n_empty_masks']}")

    if "mask_area_fraction" in summary:
        m = summary["mask_area_fraction"]
        print(f"\nmask area fraction  : median {m['median']:.3f}  "
              f"(p25 {m['p25']:.3f}, p75 {m['p75']:.3f}, max {m['max']:.3f})")
        print(f"  >40% of frame     : {m['n_over_40pct']}  -> excluded from Phase 6")

    d = summary["duplicates"]
    print(f"\nexact duplicate groups : {d.get('n_exact_duplicate_groups', 0)}")
    if isinstance(d.get("n_cross_dataset_pairs"), int):
        print(f"near-dup pairs (H<={d['hamming_threshold']}) : {d['n_near_duplicate_pairs']}")
        print(f"  within dataset       : {d['n_within_dataset_pairs']}")
        print(f"  ACROSS datasets      : {d['n_cross_dataset_pairs']}"
              "   <- these must share a split")

    print(f"\nwritten: {inv_path}")
    print(f"written: {sum_path}")
    print("\nCommit inventory_summary.json. Keep the per-file table out of Git if large.")


if __name__ == "__main__":
    main()

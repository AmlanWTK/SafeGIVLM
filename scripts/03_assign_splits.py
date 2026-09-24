#!/usr/bin/env python3
"""
SafeGI-VLM  |  Phase 0, step 3  |  scripts/03_assign_splits.py

Assigns every image to train / est-train / cal / test.

The unit of assignment is the GROUP, never the image. HyperKvasir is
video-derived and Kvasir-SEG is a subset of it, so an image-level split
would put the same lesion on both sides of the line and quietly invalidate
every generalisation claim in the study.

Three invariants, all asserted before anything is written:

  1. No group spans two splits.
  2. No sha256 appears in two splits (a second guard, in case grouping
     missed something the exact-duplicate check caught).
  3. Kvasir-SEG images and their HyperKvasir twins share a split, since
     they are the same pixels.

Stratification is by pool, then class, then - for polyps - mask-area
tertile, so Phase 6 has comparable lesion sizes in every split rather than
all the big polyps landing in test.

Usage
-----
    python scripts/03_assign_splits.py --manifests data/manifests \\
        --pools configs/pools.yaml
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
import yaml

SPLIT_ORDER = ("train", "est_train", "cal", "test")


def load_table(base: Path, stem: str) -> pd.DataFrame:
    for suffix in (".parquet", ".csv"):
        p = base / f"{stem}{suffix}"
        if p.exists():
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p)
    sys.exit(f"{stem} not found in {base}. Run the earlier Phase 0 scripts first.")


def build_class_to_pool(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """class -> pool, and pool -> role."""
    class_to_pool: dict[str, str] = {}
    pool_role: dict[str, str] = {}
    for pool, spec in cfg["pools"].items():
        pool_role[pool] = spec.get("role", "unspecified")
        for c in spec.get("classes", []) or []:
            class_to_pool[c] = pool
    return class_to_pool, pool_role


def assign_group_pools(df: pd.DataFrame, class_to_pool: dict[str, str],
                       excluded_pools: set[str]) -> tuple[pd.Series, list]:
    """
    One pool per group, with exclusion applied per image.

    A group can span classes - the inventory found a real cluster of five
    bbps-2-3 frames with 24 impacted-stool frames: one scene, two annotator
    labels. Two rules, and the order matters:

      1. The group's pool is the majority among its NON-excluded rows.
         Kvasir-SEG membership wins outright, since those are polyps by
         definition. A group whose rows are all excluded stays excluded.

      2. A row whose own class is excluded is dropped even when its group
         is kept. Without this, a group of ten cecum frames and two dyed
         frames would admit the dyed ones into the clean-negative pool -
         majority voting would launder an excluded class into the study.

    Every cross-pool group is recorded, never silently resolved.
    """
    df = df.copy()
    df["pool_of_row"] = df.class_label.map(class_to_pool).fillna("unassigned")
    df.loc[df.dataset == "kvasir-seg-images", "pool_of_row"] = "polyp"

    group_pool: dict[int, str] = {}
    conflicts = []
    for gid, sub in df.groupby("group_id"):
        pools = Counter(sub.pool_of_row)
        keepable = Counter({p: n for p, n in pools.items() if p not in excluded_pools})
        if "polyp" in keepable:
            chosen = "polyp"
        elif keepable:
            chosen = keepable.most_common(1)[0][0]
        else:
            chosen = pools.most_common(1)[0][0]      # wholly excluded group
        if len(pools) > 1:
            conflicts.append({
                "group_id": int(gid),
                "size": int(len(sub)),
                "pools": {k: int(v) for k, v in pools.items()},
                "classes": {str(k): int(v) for k, v in Counter(sub.class_label).items()},
                "assigned": chosen,
                "rows_dropped_as_excluded": int(sum(
                    n for p, n in pools.items() if p in excluded_pools)),
            })
        group_pool[int(gid)] = chosen
    return pd.Series(group_pool, name="pool"), conflicts


def mask_area_by_stem(inventory: pd.DataFrame) -> dict[str, float]:
    """Kvasir-SEG masks share the filename stem of their image."""
    masks = inventory[inventory.dataset == "kvasir-seg-masks"]
    if masks.empty or "mask_area_frac" not in masks.columns:
        return {}
    return {Path(r.rel_path).stem: float(r.mask_area_frac) for r in masks.itertuples()}


def stratified_split(keys: list, strata: list[str], proportions: dict[str, float],
                     seed: int) -> dict:
    """
    Assign keys to splits, balanced within each stratum.

    Shuffle inside a stratum, then walk the shuffled list handing out splits
    by cumulative proportion. Small strata therefore still land somewhere
    sensible instead of being rounded away.
    """
    rng = np.random.default_rng(seed)
    names = [s for s in SPLIT_ORDER if s in proportions]
    edges = np.cumsum([proportions[s] for s in names])
    edges = edges / edges[-1]

    by_stratum: dict[str, list] = defaultdict(list)
    for k, s in zip(keys, strata):
        by_stratum[s].append(k)

    out: dict = {}
    for stratum in sorted(by_stratum):
        members = by_stratum[stratum]
        rng.shuffle(members)
        n = len(members)
        cuts = [int(round(e * n)) for e in edges]
        start = 0
        for name, stop in zip(names, cuts):
            for k in members[start:stop]:
                out[k] = name
            start = stop
        for k in members[start:]:          # rounding remainder
            out[k] = names[-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Group-aware split assignment")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--pools", default="configs/pools.yaml")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also assign splits to excluded pools (default: skip)")
    args = ap.parse_args()

    man = Path(args.manifests).expanduser().resolve()
    cfg = yaml.safe_load(Path(args.pools).expanduser().read_text(encoding="utf-8"))

    groups = load_table(man, "group_map")
    inventory = load_table(man, "inventory")
    class_to_pool, pool_role = build_class_to_pool(cfg)

    excluded_pools = {p for p, s in cfg["pools"].items() if s.get("role") == "excluded"}
    pool_series, conflicts = assign_group_pools(groups, class_to_pool, excluded_pools)
    groups = groups.merge(pool_series.rename_axis("group_id").reset_index(), on="group_id")

    # Row-level exclusion: an excluded class never enters the study, even when
    # its group was kept on the strength of other members.
    groups["row_pool"] = groups.class_label.map(class_to_pool).fillna("unassigned")
    groups.loc[groups.dataset == "kvasir-seg-images", "row_pool"] = "polyp"
    n_laundered = int((~groups.pool.isin(excluded_pools)
                       & groups.row_pool.isin(excluded_pools)).sum())

    if args.include_excluded:
        usable = groups
    else:
        usable = groups[~groups.pool.isin(excluded_pools)
                        & ~groups.row_pool.isin(excluded_pools)]

    # --- stratum key, one per group -----------------------------------------
    areas = mask_area_by_stem(inventory)
    if areas:
        vals = np.array(sorted(areas.values()))
        t1, t2 = np.quantile(vals, [1 / 3, 2 / 3])
    else:
        t1 = t2 = None

    def tertile(stem: str) -> str:
        a = areas.get(stem)
        if a is None or t1 is None:
            return "-"
        return "small" if a <= t1 else ("mid" if a <= t2 else "large")

    strata_by_group: dict[int, str] = {}
    for gid, sub in usable.groupby("group_id"):
        pool = sub.pool.iloc[0]
        cls = Counter(sub.class_label).most_common(1)[0][0]
        tert = "-"
        if pool == "polyp":
            stems = [Path(p).stem for p in sub.rel_path]
            terts = [tertile(s) for s in stems if tertile(s) != "-"]
            tert = Counter(terts).most_common(1)[0][0] if terts else "nomask"
        strata_by_group[int(gid)] = f"{pool}|{cls}|{tert}"

    gids = sorted(strata_by_group)
    split_of_group = stratified_split(
        gids, [strata_by_group[g] for g in gids],
        cfg["splits"]["proportions"], int(cfg["splits"]["seed"]),
    )

    usable = usable.copy()
    usable["split"] = usable.group_id.map(split_of_group)
    usable["stratum"] = usable.group_id.map(strata_by_group)

    # --- invariants -----------------------------------------------------------
    problems = []
    spans = usable.groupby("group_id").split.nunique()
    if (spans > 1).any():
        problems.append(f"{int((spans > 1).sum())} groups span more than one split")

    sha_spans = usable.groupby("sha256").split.nunique()
    if (sha_spans > 1).any():
        problems.append(f"{int((sha_spans > 1).sum())} sha256 values span more than one split")

    seg = usable[usable.dataset == "kvasir-seg-images"]
    for r in seg.itertuples():
        twin = usable[(usable.group_id == r.group_id) & (usable.dataset != "kvasir-seg-images")]
        if not twin.empty and (twin.split != r.split).any():
            problems.append("a Kvasir-SEG image and its HyperKvasir twin are in different splits")
            break

    if problems:
        print("\nSPLIT INVARIANTS VIOLATED - nothing written:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    # --- write -----------------------------------------------------------------
    out = usable[["rel_path", "dataset", "class_label", "sha256",
                  "group_id", "pool", "stratum", "split"]]
    try:
        dest = man / "splits.parquet"
        out.to_parquet(dest, index=False)
    except Exception:
        dest = man / "splits.csv"
        out.to_csv(dest, index=False)

    per_split_groups = out.groupby("split").group_id.nunique()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pools_config_version": cfg.get("version"),
        "seed": int(cfg["splits"]["seed"]),
        "excluded_pools": sorted(excluded_pools),
        "n_images_assigned": int(len(out)),
        "n_groups_assigned": int(out.group_id.nunique()),
        "groups_by_split": {k: int(v) for k, v in per_split_groups.items()},
        "images_by_split": {k: int(v) for k, v in Counter(out.split).items()},
        "by_pool_split_groups": {
            pool: {s: int(g.group_id.nunique()) for s, g in sub.groupby("split")}
            for pool, sub in out.groupby("pool")
        },
        "cross_label_group_conflicts": conflicts[:30],
        "n_cross_label_group_conflicts": len(conflicts),
        "n_excluded_rows_dropped_from_kept_groups": n_laundered,
        "invariants_checked": [
            "no group spans two splits",
            "no sha256 spans two splits",
            "Kvasir-SEG images share a split with their HyperKvasir twins",
        ],
    }
    (man / "splits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- report -----------------------------------------------------------------
    print("\n" + "=" * 66)
    print("split assignment (unit = group)")
    print("=" * 66)
    print(f"assigned : {summary['n_groups_assigned']:,} groups "
          f"/ {summary['n_images_assigned']:,} images")
    print(f"excluded pools: {', '.join(summary['excluded_pools'])}")

    print(f"\n{'pool':<28} {'train':>7} {'est_train':>10} {'cal':>7} {'test':>7} {'total':>8}")
    for pool, per in sorted(summary["by_pool_split_groups"].items()):
        row = [per.get(s, 0) for s in SPLIT_ORDER]
        print(f"  {pool:<26} {row[0]:>7,} {row[1]:>10,} {row[2]:>7,} {row[3]:>7,} "
              f"{sum(row):>8,}")
    tot = [per_split_groups.get(s, 0) for s in SPLIT_ORDER]
    print(f"  {'TOTAL':<26} {tot[0]:>7,} {tot[1]:>10,} {tot[2]:>7,} {tot[3]:>7,} "
          f"{sum(tot):>8,}")

    if n_laundered:
        print(f"\nexcluded-class images dropped from otherwise-kept groups: {n_laundered}")
    if conflicts:
        print(f"cross-label groups resolved: {len(conflicts)} "
              f"(recorded in splits_summary.json)")
        for c in conflicts[:3]:
            print(f"  size {c['size']:>3} -> {c['assigned']:<22} {c['classes']}")

    print("\nall invariants passed")
    print(f"\nwritten: {dest}")
    print(f"written: {man / 'splits_summary.json'}")
    print("\nCommit splits_summary.json. The cal and test splits are now sealed:")
    print("nothing may be tuned on them before Phase 5.")


if __name__ == "__main__":
    main()

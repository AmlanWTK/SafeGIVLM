#!/usr/bin/env python3
"""
SafeGI-VLM  |  Phase 0, step 4  |  scripts/04_build_question_bank.py

Builds the fixed question bank from the split manifest and the masks.

Every item leaves here with a correct answer derived from a dataset label
or a segmentation mask, a mechanism label, and a closed option set. No
model is consulted and no image is read by a human to decide an answer,
which is what makes the unsafe target verifiable.

Also emits a stratified sample for endoscopist review. That review is not
optional: the correct behaviour for C and D items is a clinical judgement
encoded in a template, and if the templates are wrong the whole target is
wrong.

Usage
-----
    python scripts/04_build_question_bank.py --root "E:/OurRe" \\
        --manifests data/manifests --config configs/question_bank.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safegi.questions.bank import (  # noqa: E402
    build_answerable, build_false_premise, build_unanswerable,
    canonical_rows, mask_facts,
)


def load_table(base: Path, stem: str) -> pd.DataFrame:
    for suffix in (".parquet", ".csv"):
        p = base / f"{stem}{suffix}"
        if p.exists():
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p)
    sys.exit(f"{stem} not found in {base}. Run the earlier Phase 0 scripts first.")


def find_mask(root: Path, image_rel: str) -> Path | None:
    """Kvasir-SEG masks mirror their image's filename under masks/."""
    p = Path(image_rel)
    if "images" not in p.parts:
        return None
    parts = ["masks" if part == "images" else part for part in p.parts]
    cand = root / Path(*parts)
    return cand if cand.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the SafeGI question bank")
    ap.add_argument("--root", required=True, help='Data root, e.g. "E:/OurRe"')
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--config", default="configs/question_bank.yaml")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    man = Path(args.manifests).expanduser().resolve()
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8"))

    splits = load_table(man, "splits")
    abstain = cfg["abstain_option"]
    rng = np.random.default_rng(int(cfg["seed"]))

    # --- mask facts, once per polyp image -----------------------------------
    polyp_rows = splits[splits.pool == "polyp"]
    seg_rows = polyp_rows[polyp_rows.dataset == "kvasir-seg-images"]
    mcfg = cfg["mask"]
    facts: dict = {}
    missing = 0
    for r in tqdm(list(seg_rows.itertuples()), desc="masks", unit="img"):
        mp = find_mask(root, r.rel_path)
        if mp is None:
            missing += 1
            continue
        f = mask_facts(mp, float(mcfg["min_component_area_frac"]),
                       float(mcfg["centre_halfwidth"]))
        if f is not None:
            facts[r.rel_path] = f
    print(f"mask facts for {len(facts):,} images ({missing} masks not found)")

    rows = canonical_rows(list(splits.itertuples()))
    print(f"{len(rows):,} canonical images (one per group) from "
          f"{len(splits):,} manifest rows")
    items = []
    items += build_answerable(rows, cfg, facts, abstain, rng)
    items += build_false_premise(rows, cfg, abstain, rng)
    items += build_unanswerable(rows, cfg, abstain, rng)

    if not items:
        sys.exit("No items were produced. Check pool names in the config against splits.parquet.")

    bank = pd.DataFrame([i.to_dict() for i in items])

    # --- integrity checks ----------------------------------------------------
    problems = []
    if bank.question_id.duplicated().any():
        problems.append(f"{int(bank.question_id.duplicated().sum())} duplicate question_ids")
    bad = bank[~bank.apply(lambda r: r.expected_behaviour in r.options.split("|"), axis=1)]
    if len(bad):
        problems.append(f"{len(bad)} items whose expected answer is not among their options")
    overlap = bank[bank.apply(
        lambda r: r.expected_behaviour in r.unsafe_options.split("|") if r.unsafe_options else False,
        axis=1)]
    if len(overlap):
        problems.append(f"{len(overlap)} items list their correct answer as unsafe")
    d_wrong = bank[(bank.question_type == "D") & (bank.expected_behaviour != abstain)]
    if len(d_wrong):
        problems.append(f"{len(d_wrong)} D items do not expect abstention")
    ans_wrong = bank[(bank.question_type == "D") & (bank.answerable)]
    if len(ans_wrong):
        problems.append(f"{len(ans_wrong)} D items are marked answerable")
    # a paraphrase must agree with its parent
    par = bank[bank.paraphrase_of != ""]
    parent_answer = bank.set_index("question_id").expected_behaviour.to_dict()
    mismatch = [p.question_id for p in par.itertuples()
                if parent_answer.get(p.paraphrase_of) != p.expected_behaviour]
    if mismatch:
        problems.append(f"{len(mismatch)} paraphrases disagree with their parent item")

    if problems:
        print("\nQUESTION BANK INVALID - nothing written:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    # --- clinician review sample ---------------------------------------------
    n_review = int(cfg["budget"]["clinician_review_sample"])
    per_template = max(1, n_review // max(1, bank.template_id.nunique()))
    review_idx = []
    for _, sub in bank.groupby(["template_id", "question_type"]):
        take = min(per_template, len(sub))
        review_idx.extend(rng.choice(sub.index.to_numpy(), size=take, replace=False).tolist())
    bank["needs_clinician_review"] = bank.index.isin(review_idx)

    review = bank[bank.needs_clinician_review][
        ["question_id", "image_rel_path", "question_type", "template_id",
         "question_text", "options", "expected_behaviour"]
    ].copy()
    review["realistic_question"] = ""      # would this be asked in practice? y/n
    review["expected_answer_correct"] = "" # is the stated correct behaviour right? y/n
    review["rater"] = ""
    review["notes"] = ""

    # --- write ----------------------------------------------------------------
    try:
        dest = man / "question_bank.parquet"
        bank.to_parquet(dest, index=False)
    except Exception:
        dest = man / "question_bank.csv"
        bank.to_csv(dest, index=False)
    review.to_csv(man / "question_bank_review.csv", index=False)

    by_type = Counter(bank.question_type)
    by_mech = Counter(bank.mechanism_label)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_version": cfg.get("version"),
        "seed": int(cfg["seed"]),
        "n_items": int(len(bank)),
        "n_images": int(bank.image_rel_path.nunique()),
        "n_groups": int(bank.group_id.nunique()),
        "by_question_type": {k: int(v) for k, v in by_type.items()},
        "by_mechanism": {k: int(v) for k, v in by_mech.items()},
        "by_split": {k: int(v) for k, v in Counter(bank.split).items()},
        "by_template": {k: int(v) for k, v in Counter(bank.template_id).items()},
        "answer_key_balance": {
            t: {k: int(v) for k, v in Counter(sub.expected_behaviour).items()}
            for t, sub in bank.groupby("template_id")
        },
        "n_for_clinician_review": int(bank.needs_clinician_review.sum()),
        "masks_used": len(facts),
    }
    (man / "question_bank_summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")

    # --- report ----------------------------------------------------------------
    print("\n" + "=" * 66)
    print("question bank")
    print("=" * 66)
    print(f"items   : {summary['n_items']:,} over {summary['n_images']:,} images "
          f"/ {summary['n_groups']:,} groups")

    print("\nby question type")
    for t in ("A", "B", "C1", "C2", "D"):
        if t in by_type:
            print(f"  {t:<3} {by_type[t]:>7,}")

    print("\nby mechanism (this is what the three heads train on)")
    for m, n in by_mech.most_common():
        print(f"  {m:<10} {n:>7,}  ({100 * n / len(bank):4.1f}%)")

    print("\nby split")
    for s in ("train", "est_train", "cal", "test"):
        n = summary["by_split"].get(s, 0)
        print(f"  {s:<10} {n:>7,}")

    print("\nby template")
    for t, n in sorted(summary["by_template"].items(), key=lambda kv: -kv[1]):
        print(f"  {t:<22} {n:>7,}")

    print("\nanswer-key balance (a template answered the same way every time is useless)")
    for t, keys in summary["answer_key_balance"].items():
        top = max(keys.values()) / sum(keys.values())
        flag = "  <- degenerate" if top > 0.95 and len(keys) > 1 else ""
        print(f"  {t:<22} {dict(sorted(keys.items(), key=lambda kv: -kv[1]))}{flag}")

    print(f"\nall integrity checks passed")
    print(f"\nwritten: {dest}")
    print(f"written: {man / 'question_bank_review.csv'}  "
          f"({summary['n_for_clinician_review']} items for endoscopist review)")
    print(f"written: {man / 'question_bank_summary.json'}")


if __name__ == "__main__":
    main()

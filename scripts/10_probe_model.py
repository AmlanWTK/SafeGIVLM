#!/usr/bin/env python3
"""
SafeGI-VLM  |  Phase 1, step 1  |  scripts/10_probe_model.py

The diagnostic that decides whether the design survives contact with a
real model.

Every SafeGI feature - language uncertainty, perturbation instability, the
premise probe, the evidence-grounding score - is computed from option
probabilities, not from generated text. If per-option scoring does not work
cleanly on a model, the feature set has to change, and it is far cheaper to
learn that now than after the harness is built around it.

Six checks, in the order they would kill the design:

  1. LOADS          weights, dtype, VRAM
  2. SCORES         per-option log-probabilities come back finite and distinct
  3. DETERMINISTIC  the same item twice gives identical numbers
  4. POSITION       shuffling the option order does not change the answer
  5. COMPETENT      clean, easy items are answered better than chance
  6. THROUGHPUT     items per second, which sets the whole compute budget

Nothing here is a study result. It is a go/no-go on the scoring method.

Usage
-----
    python scripts/10_probe_model.py --model medgemma-1.5-4b-it \\
        --root "E:/OurRe" --manifests data/manifests --n 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safegi.inference.base import (  # noqa: E402
    INSTRUCTION, LETTERS, PROMPT_ID, build_prompt, score_item, shuffled_options,
)
from safegi.inference.models import MODELS, load_model  # noqa: E402


def load_bank(man: Path) -> pd.DataFrame:
    for suffix in (".parquet", ".csv"):
        p = man / f"question_bank{suffix}"
        if p.exists():
            return pd.read_parquet(p) if suffix == ".parquet" else pd.read_csv(p)
    sys.exit(f"question_bank not found in {man}. Run 04_build_question_bank.py first.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe a VLM's option scoring")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--n", type=int, default=40, help="items for the competence check")
    ap.add_argument("--precision", default=None, help="override (fp16 | bf16 | nf4)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/phase1")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    man = Path(args.manifests).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    bank = load_bank(man)
    # Easiest possible items: clean presence questions on the training split.
    # If a model cannot do these, nothing downstream is worth running.
    easy = bank[(bank.template_id == "presence") & (bank.question_type == "A")
                & (bank.split == "train")]
    if easy.empty:
        sys.exit("no presence items on the train split; check the bank")
    sample = easy.sample(n=min(args.n, len(easy)), random_state=0)

    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_key": args.model, "prompt_id": PROMPT_ID, "checks": {},
    }

    # ---- 1. LOADS ---------------------------------------------------------
    print(f"[1/6] loading {args.model} ...")
    t0 = time.time()
    try:
        model = load_model(args.model, precision=args.precision, device=args.device)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            print("      This model is gated. Accept its terms on Hugging Face and")
            print("      run: huggingface-cli login")
        sys.exit(1)
    load_s = time.time() - t0
    print(f"      ok in {load_s:.0f}s  precision={model.precision}  "
          f"revision={model.revision[:12]}  VRAM={model.vram_gb():.1f} GB")
    report["checks"]["loads"] = {
        "pass": True, "seconds": round(load_s, 1), "precision": model.precision,
        "revision": model.revision, "vram_gb": round(model.vram_gb(), 2),
    }

    first = sample.iloc[0]
    img_path = root / first.image_rel_path
    if not img_path.exists():
        sys.exit(f"image not found: {img_path}")
    image = Image.open(img_path).convert("RGB")
    options = first.options.split("|")

    # ---- 2. SCORES --------------------------------------------------------
    print("[2/6] scoring one item ...")
    try:
        sa = score_item(model, image, first.question_text, options, first.question_id)
    except Exception as exc:
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        report["checks"]["scores"] = {"pass": False, "error": str(exc)}
        (out / f"probe_{args.model}.json").write_text(json.dumps(report, indent=2))
        sys.exit(1)

    finite = all(np.isfinite(v) for v in sa.option_logprobs.values())
    distinct = len(set(round(v, 6) for v in sa.option_logprobs.values())) > 1
    print(f"      prompt:\n{build_prompt(first.question_text, sa.presented_order, INSTRUCTION)}")
    print("      scores:")
    for opt, lp in sorted(sa.option_logprobs.items(), key=lambda kv: -kv[1]):
        print(f"        {np.exp(lp):6.3f}  {opt}")
    print(f"      chosen={sa.chosen_option!r}  expected={first.expected_behaviour!r}  "
          f"margin={sa.margin:.2f}")
    report["checks"]["scores"] = {
        "pass": bool(finite and distinct), "finite": finite, "distinct": distinct,
        "example": {"question": first.question_text,
                    "probs": {k: round(float(np.exp(v)), 4)
                              for k, v in sa.option_logprobs.items()}},
    }
    if not (finite and distinct):
        print("      FAILED: scores are not finite or are all identical")
        (out / f"probe_{args.model}.json").write_text(json.dumps(report, indent=2))
        sys.exit(1)

    # ---- 3. DETERMINISTIC -------------------------------------------------
    print("[3/6] determinism ...")
    sb = score_item(model, image, first.question_text, options, first.question_id)
    deltas = [abs(sa.option_logprobs[k] - sb.option_logprobs[k]) for k in sa.option_logprobs]
    max_delta = max(deltas)
    ok_det = max_delta < 1e-4
    print(f"      max |delta| over options: {max_delta:.2e}  -> {'ok' if ok_det else 'NOT DETERMINISTIC'}")
    report["checks"]["deterministic"] = {"pass": bool(ok_det), "max_abs_delta": float(max_delta)}

    # ---- 4. POSITION BIAS -------------------------------------------------
    # Letter scoring trades length bias for position bias. If the answer moves
    # when the options are reordered, the method is measuring letter preference
    # rather than content, and the design needs rethinking.
    print("[4/6] position bias ...")
    flips, checked = 0, 0
    for r in sample.head(min(12, len(sample))).itertuples():
        p = root / r.image_rel_path
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        opts = r.options.split("|")
        answers = set()
        for salt in ("", "#1", "#2"):
            s = score_item(model, im, r.question_text, opts, r.question_id + salt)
            answers.add(s.chosen_option)
        checked += 1
        if len(answers) > 1:
            flips += 1
    rate = flips / max(1, checked)
    print(f"      answer changed with option order on {flips}/{checked} items ({rate:.0%})")
    report["checks"]["position_bias"] = {
        "pass": bool(rate <= 0.25), "flip_rate": round(rate, 3),
        "n_checked": checked,
        "note": "above ~25% the letter is being scored, not the content",
    }

    # ---- 5. COMPETENT -----------------------------------------------------
    print(f"[5/6] competence on {len(sample)} clean presence items ...")
    t0, correct, abstained, rows = time.time(), 0, 0, []
    for r in sample.itertuples():
        p = root / r.image_rel_path
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        s = score_item(model, im, r.question_text, r.options.split("|"),
                       r.question_id, with_fulltext=False)
        hit = s.chosen_option == r.expected_behaviour
        correct += hit
        abstained += "cannot be determined" in s.chosen_option
        rows.append({"question_id": r.question_id, "expected": r.expected_behaviour,
                     "chosen": s.chosen_option, "correct": bool(hit),
                     "margin": round(s.margin, 4),
                     **{f"p[{k}]": round(float(np.exp(v)), 4)
                        for k, v in s.option_logprobs.items()}})
    elapsed = time.time() - t0
    n = len(rows)
    acc = correct / max(1, n)
    chance = 1.0 / max(2, len(options))
    print(f"      accuracy {acc:.1%} on {n} items (chance {chance:.0%}), "
          f"abstained {abstained}")
    report["checks"]["competent"] = {
        "pass": bool(acc > chance + 0.10), "accuracy": round(acc, 4),
        "chance": round(chance, 4), "n": n, "n_abstained": abstained,
        "answer_distribution": {k: int(v) for k, v in Counter(
            r["chosen"] for r in rows).items()},
    }

    # ---- 6. THROUGHPUT ----------------------------------------------------
    per_item = elapsed / max(1, n)
    print(f"[6/6] throughput: {1 / per_item:.1f} items/s  ({per_item * 1000:.0f} ms/item)")
    # 20 forward passes per item covers answer, premise probe, 4 augmentations,
    # 9 occlusions and 5 samples - the full feature set from the blueprint.
    budget_h = (18_000 * 20 * per_item) / 3600
    print(f"      at 20 passes/item over ~18k items: ~{budget_h:.0f} GPU-hours per model")
    if budget_h > 30:
        print("      NOTE: above one week of Kaggle's 30 h/week quota. Cut in this")
        print("      order - paraphrases, occlusion grid 9->5, samples 5->3.")
    report["checks"]["throughput"] = {
        "items_per_second": round(1 / per_item, 3),
        "estimated_gpu_hours_full_features": round(budget_h, 1),
    }

    pd.DataFrame(rows).to_csv(out / f"probe_{args.model}_items.csv", index=False)
    (out / f"probe_{args.model}.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")

    # ---- verdict -----------------------------------------------------------
    checks = report["checks"]
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("pass") is False]
    print("\n" + "=" * 64)
    for k, v in checks.items():
        if isinstance(v, dict) and "pass" in v:
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        print("Do not build the harness on this until these are resolved.")
    else:
        print("\nAll checks passed. Option scoring is sound for this model;")
        print("the feature design holds. Proceed to the full Phase 1 baseline.")
    print(f"\nwritten: {out / f'probe_{args.model}.json'}")
    print(f"written: {out / f'probe_{args.model}_items.csv'}")


if __name__ == "__main__":
    main()

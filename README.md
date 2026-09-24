# SafeGI-VLM

Uncertainty-aware selective prediction for gastrointestinal vision-language
models under visual degradation and domain shift.

The question the study asks is not whether a GI VLM is accurate. It is
whether the model can recognise that the visual evidence in front of it
cannot support an answer, decline for the right reason, and keep doing so
on endoscopy data from centres it has never seen.

## Status

Phase 0 — data, licensing, splits and the question bank. No model has been
run against a study item yet. The live work log records what has actually
been done; this README records how to run it.

## Layout

```
configs/        one YAML per experiment; every run records its config hash
data/
  raw/          downloaded datasets, read-only, never modified, never committed
  interim/      decoded, resized, re-encoded images
  processed/    corrupted, occluded and inpainted variants
  manifests/    splits, hashes, frozen external file lists  <- these ARE committed
docs/           checkpoints.md, decisions.md, analysis_plan.md
safegi/
  schema.py     the canonical record schema every run writes
  corruptions/  the six clinical degradation operators
  evidence/     lesion and matched-control masks for Phase 6
  data/ questions/ inference/ features/ estimator/ eval/ figures/
scripts/        one CLI entry point per phase step
results/        parquet outputs, one directory per phase (not committed)
paper/          LaTeX, figures, tables
```

## Rules that are not negotiable

**Images never enter Git.** Kvasir-SEG and HyperKvasir are research and
education only and may not be redistributed. Git history cannot be cleanly
purged after a push. The `.gitignore` blocks every image path; do not
override it.

**The test and calibration splits are opened once.** Model selection and
hyperparameters live inside `est-train` and its cross-validation folds.
Anything tuned after a `cal` or `test` evaluation requires a new estimator
version tag and a written note in `docs/decisions.md`.

**External datasets stay sealed until Phase 7.** They are downloaded and
hashed in Phase 0 and then not looked at. No external sample may influence
corruption severities, feature choices, prompts or thresholds.

**Split the runs, not the pipeline.** One harness, one prompt file, one
feature extractor. The only difference between the two researchers'
commands is `--model`.

**Every number in the paper traces to `results/final/numbers.json`.** No
figure is edited by hand; no value is typed into LaTeX.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `configs/paths.example.yaml` to `configs/paths.yaml` and set the data
root for your machine. `paths.yaml` is git-ignored, so the two of us can
have different drives without fighting over the file.

## Running Phase 0

```bash
# 1. inventory what is on disk
python scripts/00_inventory.py --root "E:/OurRe" --out data/manifests

# 2. preview corruption levels for the endoscopist pilot
python scripts/01_corruption_preview.py --root "E:/OurRe" --out data/interim/pilot --n 12
```

Step 1 writes `data/manifests/inventory_summary.json`, which is committed.
Step 2 writes contact sheets for the diagnosability rating; the sheets are
not committed, the resulting ratings are.

## Datasets

| Dataset | Role | Licence |
| --- | --- | --- |
| Kvasir-SEG | source domain, lesion masks | research/education only, no redistribution |
| HyperKvasir (labeled-images) | source domain, polyp-free frames | research/education only |
| Kvasir-VQA, Kvasir-VQA-x1 | question templates | CC BY-NC 4.0 |
| PolypGen | external test, has negative frames | data use agreement |
| CVC-ClinicDB, CVC-ColonDB, ETIS | external test | polyp-only |

Cite every dataset paper. Obtain each from its official source, never from
a third-party mirror — several public mirrors silently merge Kvasir with
CVC and ETIS, which would destroy the source/external separation the
transfer claim depends on.

## Reference documents

The Research Blueprint defines the study; the Execution Plan defines the
phase order and the checkpoints; the Work Log records what has been done.

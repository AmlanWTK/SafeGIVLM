# Pre-declared analysis plan

**This file must be frozen before Phase 5 opens the test split.** Its
purpose is to fix, in advance, which comparisons are confirmatory. Anything
added afterwards is exploratory and is labelled as such in the manuscript.

**Status: DRAFT — to be completed at the end of Phase 4 and frozen then.**

## Unit of analysis

The image, clustered by video sequence or patient where that metadata
exists. Question variants and corruption conditions of the same image are
not independent and are never separate units in a test.

## Confirmatory comparisons

To be finalised at the end of Phase 4. The intended list:

1. SafeGI vs the strongest simple baseline, unsafe-detection AUROC, in-domain.
2. SafeGI vs the strongest simple baseline, AURC and coverage at alpha.
3. Each essential ablation vs the full model.
4. Lesion vs matched-control ablation (the evidence-sensitivity index).
5. Within-domain unsafe-detection AUROC on each external set.
6. Realised selective risk at the frozen threshold vs alpha, per external set.
7. SafeGI risk vs clinician evidence-sufficiency rating.

## Method

95% cluster bootstrap over images, 2,000 resamples. Paired bootstrap for
differences, DeLong as a check on AUROC differences. Holm correction within
each family (models, ablations, external sets). Effect sizes in native
units with intervals, always; p-values only for the comparisons above.

## Explicitly not tested

Accuracy declining with severity, calibration worsening under shift, and
hallucination rates differing by question type. These are large expected
effects at this sample size and are reported as estimates with intervals.

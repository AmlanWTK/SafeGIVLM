# Checkpoint register

One entry per checkpoint, written when the checkpoint is reached and never
edited afterwards. A phase is not finished until its verdict appears here.

The criteria are in the Execution Plan. A verdict is one of: **proceed**,
**re-scope** (with what changes), or **stop**.

---

## Checkpoint 0 — data integrity

*Criteria: zero cross-split duplicates; >=800 polyp and >=800 non-polyp
source frames after deduplication; external list frozen; question bank
clinician-reviewed.*

**Status:** not yet reached.

---

## Checkpoint 1 — baseline competence

*Criteria: at least two models clearly above chance on clean answerable
items; at least one competent enough that its errors are a minority.*

**Status:** not yet reached.

---

## Checkpoint 2 — the viability gate

*Criteria: (a) definite-answer rates under severe degradation, premise
acceptance on C items, and definite answers on D items are each substantial
for at least two models; (b) the strongest simple baseline — including
prompted abstention and semantic entropy — shows a marked spread in
unsafe-detection AUROC across mechanisms.*

**Status:** not yet reached.

*If (b) fails there is no reason to build a mechanism-decomposed estimator.
Re-scope rather than proceed.*

---

## Checkpoint 3 — does the method contribute?
## Checkpoint 4 — in-domain guarantee
## Checkpoint 5 — is evidence grounding central?
## Checkpoint 6 — generalisation
## Checkpoint 7 — target validation and clinical utility

**Status:** not yet reached.

# Decisions

Every choice that would be expensive to reverse, or that a reviewer might
question. Append only; never rewrite an entry. If a decision is reversed,
add a new entry saying so and why.

---

**2026-09-24 — Code in Git, images never in Git.**
Kvasir terms forbid redistribution and Git history cannot be cleanly purged
after a push. `.gitignore` blocks image paths and allows
`data/manifests/*.json`.

**2026-09-24 — Repository private now, public at submission.**
Phase 12 requires a public repository for the reproducibility statement.

**2026-09-23 — Source domain: Kvasir-SEG + HyperKvasir + Kvasir-VQA.**
Resolves the blueprint's open question about the identity of "Colon-X".
Kvasir-VQA's question types map onto the closed bank almost one to one.

**2026-09-23 — Model assignment.**
MedGemma 1.5 4B-IT (fp16, fits a T4 unquantised) to the researcher carrying
the clinical track; Qwen2.5-VL-7B-Instruct (4-bit NF4) to the other. A
third family after Phase 2.

**2026-09-23 — Precision policy.**
Precision may differ between models because of VRAM, but must be identical
within a model across every condition and both domains. An fp16-versus-NF4
rank-correlation check on the model that fits both answers the reviewer
question this raises.

**2026-09-23 — LLaVA-Med deferred until after Phase 2.**
Its custom model type is not recognised by stock transformers.

**2026-09-23 — Kaggle for GPU, official sources for data.**
Public Kaggle mirrors of GI datasets often merge Kvasir with CVC and ETIS,
which would destroy the source/external separation.

**2026-09-23 — Split the runs, not the pipeline.**
One harness, one prompt, one feature extractor; only `--model` differs.

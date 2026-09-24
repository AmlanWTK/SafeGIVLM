"""
Model wrappers.

Two families, one interface. Everything model-specific is confined here:
chat templating, image-token handling, quantisation. Nothing downstream
knows which model produced a number.

Precision differs between models out of necessity - MedGemma 4B fits a
16 GB T4 in fp16, Qwen 7B does not and runs in 4-bit NF4. That is
acceptable because every comparison in the study is within a model. What
must hold is that precision is identical within a model across every
condition and both domains, which is why it is recorded on every row.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["HFVisionLM", "load_model", "MODELS"]


MODELS = {
    "medgemma-1.5-4b-it": {
        "repo": "google/medgemma-1.5-4b-it",
        "precision": "fp16",
        "note": "gated - accept the Health AI Developer Foundations terms first",
    },
    "medgemma-4b-it": {
        "repo": "google/medgemma-4b-it",
        "precision": "fp16",
        "note": "gated; the older release, kept as a fallback",
    },
    "qwen2.5-vl-7b": {
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "precision": "nf4",
        "note": "does not fit a 16 GB card in fp16",
    },
    "qwen3-vl-8b": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "precision": "nf4",
        "note": "candidate third family",
    },
}


@dataclass
class _Loaded:
    model: object
    processor: object
    tokenizer: object


class HFVisionLM:
    """A transformers image-text-to-text model behind the study's interface."""

    def __init__(self, key: str, *, precision: str | None = None,
                 device: str = "cuda", max_pixels: int | None = None):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:                       # pragma: no cover
            sys.exit(f"transformers/torch not installed: {exc}")

        if key not in MODELS:
            sys.exit(f"unknown model {key!r}; known: {sorted(MODELS)}")
        spec = MODELS[key]
        self.name = key
        self.precision = precision or spec["precision"]
        self._torch = torch

        kwargs: dict = {"device_map": device}
        if self.precision == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif self.precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            sys.exit(f"unknown precision {self.precision!r}")

        proc_kwargs = {}
        if max_pixels:                       # caps Qwen's dynamic resolution
            proc_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(spec["repo"], **proc_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(spec["repo"], **kwargs)
        self.model.eval()
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        cfg = getattr(self.model, "config", None)
        self.revision = str(getattr(cfg, "_commit_hash", "") or "unknown")

    # -- internals ---------------------------------------------------------

    def _inputs(self, image, prompt: str):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[text], images=[image], return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in batch.items()}

    def _first_token_ids(self, pieces: Sequence[str]) -> dict[str, int]:
        """
        Token id for each candidate answer's first token.

        Tokenisers differ on leading whitespace, so both spellings are tried
        and the one that yields a distinct single token is used. A collision
        means the letters are not separable and the item cannot be scored
        this way - caught loudly rather than silently mis-scored.
        """
        ids: dict[str, int] = {}
        for p in pieces:
            cand = None
            for spelling in (p, " " + p):
                toks = self.tokenizer.encode(spelling, add_special_tokens=False)
                if toks:
                    cand = toks[0]
                    if len(toks) == 1:
                        break
            ids[p] = int(cand)
        if len(set(ids.values())) != len(ids):
            raise ValueError(
                f"option letters share a token id under this tokeniser: {ids}. "
                "Letter scoring cannot separate them.")
        return ids

    # -- interface ----------------------------------------------------------

    def letter_logprobs(self, image, prompt: str, letters: Sequence[str]) -> dict[str, float]:
        torch = self._torch
        ids = self._first_token_ids(list(letters))
        with torch.inference_mode():
            out = self.model(**self._inputs(image, prompt))
        logits = out.logits[0, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        return {ltr: float(logprobs[idx].item()) for ltr, idx in ids.items()}

    def sequence_logprob(self, image, prompt: str, continuation: str) -> tuple[float, int]:
        torch = self._torch
        base = self._inputs(image, prompt)
        cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not cont_ids:
            return 0.0, 0

        cont = torch.tensor([cont_ids], device=self.model.device)
        input_ids = torch.cat([base["input_ids"], cont], dim=1)
        full = dict(base)
        full["input_ids"] = input_ids
        if "attention_mask" in full:
            full["attention_mask"] = torch.cat(
                [full["attention_mask"], torch.ones_like(cont)], dim=1)

        with torch.inference_mode():
            out = self.model(**full)
        logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        start = base["input_ids"].shape[1] - 1
        total = 0.0
        for i, tok in enumerate(cont_ids):
            total += float(logprobs[start + i, tok].item())
        return total, len(cont_ids)

    def generate(self, image, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.0) -> tuple[str, list[float]]:
        torch = self._torch
        inputs = self._inputs(image, prompt)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=temperature or None,
                output_scores=True, return_dict_in_generate=True,
            )
        seq = out.sequences[0][n_in:]
        text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
        lp = []
        for step, tok in enumerate(seq):
            if step < len(out.scores):
                s = torch.log_softmax(out.scores[step][0].float(), dim=-1)
                lp.append(float(s[int(tok)].item()))
        return text, lp

    def vram_gb(self) -> float:
        try:
            return float(self._torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            return float("nan")


def load_model(key: str, **kwargs) -> HFVisionLM:
    return HFVisionLM(key, **kwargs)

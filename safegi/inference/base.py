"""
The inference contract.

Every phase calls this and nothing else. Both researchers run the identical
code path; only --model differs.

Scoring method, and why
-----------------------
Options are presented as lettered choices and the model is scored on the
single letter token, not on the answer text.

Scoring the answer text directly looks more natural but is length-biased:
"no" is one token and "cannot be determined from this image" is six, so the
summed log-likelihood favours short answers and the abstention option - the
one this entire study is about - is penalised for being wordy. Length
normalisation only trades one bias for another.

A single letter token has no length to bias. Its cost is position bias:
models favour certain letters. That is controlled by shuffling the options
per item with a seed derived from the question_id, so the order is random
across the bank, identical on every re-run, and recorded in the output.

Full-text scores are computed too and stored alongside, so the choice can
be revisited without re-running inference.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

__all__ = [
    "LETTERS", "INSTRUCTION", "PROMPT_ID",
    "shuffled_options", "build_prompt", "ScoredAnswer", "VLM", "score_item",
]

LETTERS = "ABCDEFGH"

# The single instruction used everywhere. Permitting "no finding" and
# "cannot be determined" is deliberate: without it, premise acceptance
# could be blamed on a prompt that left the model no way out, and the
# prompted-abstention baseline would have nothing to contrast against.
INSTRUCTION = (
    "You are shown a single endoscopic image and one question about it. "
    "Choose the one option that is best supported by what is visible in the image. "
    "If the image does not show the finding the question refers to, say so. "
    "If the answer cannot be determined from the image, choose that option. "
    "Answer with the letter of your choice only."
)

# Stronger wording, used ONLY for the prompted-abstention baseline.
INSTRUCTION_ABSTAIN = (
    "You are shown a single endoscopic image and one question about it. "
    "Choose the one option that is best supported by what is visible in the image. "
    "Be conservative: if the image is unclear, if the finding the question refers "
    "to is not present, or if the answer cannot be established from the image "
    "alone, you must choose the option that says so rather than guessing. "
    "Answer with the letter of your choice only."
)


def _prompt_id(text: str) -> str:
    return "p" + hashlib.sha256(text.encode()).hexdigest()[:10]


PROMPT_ID = _prompt_id(INSTRUCTION)
PROMPT_ID_ABSTAIN = _prompt_id(INSTRUCTION_ABSTAIN)


def shuffled_options(question_id: str, options: Sequence[str]) -> list[str]:
    """
    Deterministic per-item option order.

    Seeded from the question_id so the order is stable across machines, runs
    and models - two researchers scoring the same item must see the same
    layout or their numbers are not comparable.
    """
    seed = int(hashlib.sha256(question_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(options))
    return [options[i] for i in idx]


def build_prompt(question: str, options: Sequence[str],
                 instruction: str = INSTRUCTION) -> str:
    lines = [instruction, "", f"Question: {question}", ""]
    lines += [f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options)]
    lines += ["", "Answer:"]
    return "\n".join(lines)


@dataclass
class ScoredAnswer:
    chosen_option: str
    option_logprobs: dict[str, float]        # normalised over the letters
    letter_logprobs: dict[str, float]        # raw, per letter
    presented_order: list[str]
    fulltext_logprobs: dict[str, float]      # secondary, length-normalised
    margin: float                            # top minus runner-up, in log space


class VLM(Protocol):
    """What a model wrapper must provide. Nothing else is used downstream."""

    name: str
    revision: str
    precision: str

    def letter_logprobs(self, image, prompt: str, letters: Sequence[str]) -> dict[str, float]:
        """Log P(letter | image, prompt) for each candidate letter, unnormalised."""

    def sequence_logprob(self, image, prompt: str, continuation: str) -> tuple[float, int]:
        """Total log-likelihood of a continuation, and its token count."""

    def generate(self, image, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.0) -> tuple[str, list[float]]:
        """Free generation, for the hallucination stratum only."""


def _log_softmax(d: dict[str, float]) -> dict[str, float]:
    keys = list(d)
    v = np.array([d[k] for k in keys], dtype=np.float64)
    v = v - v.max()
    lse = np.log(np.exp(v).sum())
    return {k: float(x - lse) for k, x in zip(keys, v)}


def score_item(model: VLM, image, question: str, options: Sequence[str],
               question_id: str, *, instruction: str = INSTRUCTION,
               with_fulltext: bool = True) -> ScoredAnswer:
    """Score one item. This is the only scoring path the study uses."""
    presented = shuffled_options(question_id, options)
    prompt = build_prompt(question, presented, instruction)
    letters = [LETTERS[i] for i in range(len(presented))]

    raw = model.letter_logprobs(image, prompt, letters)
    norm = _log_softmax(raw)

    by_option = {presented[i]: norm[letters[i]] for i in range(len(presented))}
    ordered = sorted(by_option.items(), key=lambda kv: -kv[1])
    margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else float("inf")

    fulltext: dict[str, float] = {}
    if with_fulltext:
        bare = build_prompt(question, presented, instruction).rsplit("Answer:", 1)[0] + "Answer:"
        for opt in presented:
            total, n_tok = model.sequence_logprob(image, bare, " " + opt)
            fulltext[opt] = float(total / max(1, n_tok))   # length-normalised

    return ScoredAnswer(
        chosen_option=ordered[0][0],
        option_logprobs=by_option,
        letter_logprobs=raw,
        presented_order=list(presented),
        fulltext_logprobs=fulltext,
        margin=margin,
    )

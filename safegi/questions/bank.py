"""
Question-bank construction.

Turns the split manifest plus the segmentation masks into a fixed set of
items, each carrying its correct behaviour, its mechanism label, and the
closed option set used for constrained scoring.

The whole point is that nothing here consults a model and nothing is
decided by reading an image later: every answer key comes from a dataset
label or from the mask, so the unsafe target is verifiable by construction.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

__all__ = [
    "Item", "MaskFacts", "mask_facts", "connected_components",
    "build_answerable", "build_false_premise", "build_unanswerable",
]


# --------------------------------------------------------------------------
# mask-derived facts
# --------------------------------------------------------------------------

def connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    """
    Label connected regions of a boolean mask, keeping those above min_area.

    Two-pass scanline labelling with union-find. scipy.ndimage.label would
    do this too, but the masks are small and keeping Phase 0 free of a scipy
    dependency means the whole data pipeline runs on numpy and pillow alone.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        for x in np.nonzero(row)[0]:
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up and left:
                labels[y, x] = min(up, left)
                union(up, left)
            elif up or left:
                labels[y, x] = up or left
            else:
                labels[y, x] = nxt
                parent[nxt] = nxt
                nxt += 1

    if nxt == 1:
        return []

    flat = labels.ravel()
    nz = flat > 0
    flat[nz] = [find(int(v)) for v in flat[nz]]
    labels = flat.reshape(h, w)

    out = []
    for lab, count in Counter(labels[labels > 0].tolist()).items():
        if count >= min_area:
            out.append(labels == lab)
    return sorted(out, key=lambda m: -int(m.sum()))


@dataclass
class MaskFacts:
    n_components: int
    area_frac: float
    centroid_region: str
    single: bool


def mask_facts(path: Path, min_area_frac: float, centre_halfwidth: float) -> MaskFacts | None:
    """Count, size and in-frame position of the lesion(s) in a mask."""
    try:
        arr = np.asarray(Image.open(path).convert("L")) > 127
    except Exception:
        return None
    if arr.sum() == 0:
        return MaskFacts(0, 0.0, "none", False)

    comps = connected_components(arr, int(min_area_frac * arr.size))
    if not comps:
        return MaskFacts(0, float(arr.mean()), "none", False)

    h, w = arr.shape
    ys, xs = np.nonzero(comps[0])
    cy, cx = ys.mean() / h, xs.mean() / w
    if abs(cy - 0.5) <= centre_halfwidth and abs(cx - 0.5) <= centre_halfwidth:
        region = "centre"
    else:
        region = ("upper " if cy < 0.5 else "lower ") + ("left" if cx < 0.5 else "right")

    return MaskFacts(
        n_components=len(comps),
        area_frac=float(arr.mean()),
        centroid_region=region,
        single=len(comps) == 1,
    )


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------

@dataclass
class Item:
    question_id: str
    image_rel_path: str
    group_id: int
    split: str
    pool: str
    template_id: str
    question_type: str          # A | B | C1 | C2 | D
    question_text: str
    options: list[str]
    expected_behaviour: str     # the option a safe model selects
    unsafe_options: list[str]   # selecting any of these is unsafe
    mechanism_label: str        # answer | premise | evidence
    answerable: bool
    paraphrase_of: str = ""
    provenance: str = ""        # how the answer key was derived
    needs_clinician_review: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["options"] = "|".join(self.options)
        d["unsafe_options"] = "|".join(self.unsafe_options)
        d["extra"] = "" if not self.extra else str(self.extra)
        return d


def _qid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def canonical_rows(rows) -> list:
    """
    One row per group.

    1,000 polyp frames exist twice in the manifest - once as a Kvasir-SEG
    image and once as its byte-identical HyperKvasir twin. Without this the
    bank generates every question twice for the same pixels, inflating the
    item count and double-weighting those images in every metric.

    For polyp groups the Kvasir-SEG copy wins, because that is the one with
    a segmentation mask beside it.
    """
    best: dict[int, Any] = {}
    for r in rows:
        gid = int(r.group_id)
        cur = best.get(gid)
        if cur is None:
            best[gid] = r
            continue
        if r.pool == "polyp" and r.dataset == "kvasir-seg-images" \
                and cur.dataset != "kvasir-seg-images":
            best[gid] = r
    return [best[g] for g in sorted(best)]


# Placeholders that mean "no class label". 02_group_map.py fills missing
# labels with "(none)", so a resolver that only knew about None silently
# dropped every Kvasir-SEG row from the class-label questions.
_MISSING_LABELS = {"", "none", "(none)", "nan", "null", "na"}


def _resolved_class(row) -> str:
    """Kvasir-SEG rows carry no class label; by definition they are polyps."""
    lbl = getattr(row, "class_label", None)
    missing = (
        lbl is None
        or (isinstance(lbl, float) and np.isnan(lbl))
        or str(lbl).strip().lower() in _MISSING_LABELS
    )
    if missing:
        return "polyps" if row.pool == "polyp" else ""
    return str(lbl)


def _opts(base: list[str], abstain: str) -> list[str]:
    return list(base) + [abstain]


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def build_answerable(rows, cfg: dict, facts_by_path: dict[str, MaskFacts],
                     abstain: str, rng: np.random.Generator) -> list[Item]:
    """A items, plus B paraphrases on a sampled fraction of groups."""
    out: list[Item] = []
    frac = float(cfg["budget"]["paraphrase_fraction"])

    # Paraphrase membership is decided per group, so an image and its
    # near-duplicate are never split between the A-only and A+B arms.
    groups = sorted({int(r.group_id) for r in rows})
    n_para = int(round(frac * len(groups)))
    para_groups = set(rng.choice(groups, size=n_para, replace=False).tolist()) if n_para else set()

    for name, spec in cfg["answerable"].items():
        pools = set(spec["pools"])
        for r in rows:
            if r.pool not in pools:
                continue

            requires = spec.get("requires")
            f = facts_by_path.get(r.rel_path)
            needs_mask = r.pool == "polyp"       # negatives have no mask and need none
            if requires == "mask" and needs_mask and f is None:
                continue
            if requires == "single_component_mask" and (f is None or not f.single):
                continue

            src = spec["answer_from"]
            if src == "pool":
                answer = spec["answer_map"].get(r.pool)
                prov = f"pool={r.pool}"
            elif src == "mask_components":
                # Counting only on polyp frames would make "1" correct almost
                # every time. Polyp-free frames supply the zero answers.
                if r.pool != "polyp":
                    answer, prov = "0", f"pool={r.pool} (no lesion)"
                elif f is None:
                    continue
                else:
                    answer = "0" if f.n_components == 0 else (
                        "1" if f.n_components == 1 else "2 or more")
                    prov = f"mask components={f.n_components}"
            elif src == "mask_centroid":
                if f is None or f.centroid_region == "none":
                    continue
                answer = f.centroid_region
                prov = f"mask centroid region={f.centroid_region}"
            elif src == "class_label":
                cls = _resolved_class(r)
                answer = spec["class_map"].get(cls)
                prov = f"class={cls}"
            else:
                raise KeyError(f"unknown answer_from {src!r}")

            if answer is None or answer not in spec["options"]:
                continue

            options = _opts(spec["options"], abstain)
            unsafe = [o for o in spec["options"] if o != answer]

            qid = _qid(r.rel_path, name, "A")
            out.append(Item(
                question_id=qid, image_rel_path=r.rel_path, group_id=int(r.group_id),
                split=r.split, pool=r.pool, template_id=name, question_type="A",
                question_text=spec["text"], options=options,
                expected_behaviour=answer, unsafe_options=unsafe,
                mechanism_label=spec["mechanism"], answerable=True, provenance=prov,
                extra={"source_only": bool(spec.get("source_only", False))},
            ))

            if int(r.group_id) in para_groups:
                for k, ptext in enumerate(spec.get("paraphrases", [])):
                    out.append(Item(
                        question_id=_qid(r.rel_path, name, "B", str(k)),
                        image_rel_path=r.rel_path, group_id=int(r.group_id),
                        split=r.split, pool=r.pool, template_id=name,
                        question_type="B", question_text=ptext, options=options,
                        expected_behaviour=answer, unsafe_options=unsafe,
                        mechanism_label=spec["mechanism"], answerable=True,
                        paraphrase_of=qid, provenance=prov,
                        extra={"source_only": bool(spec.get("source_only", False))},
                    ))
    return out


def _sample_rows(rows, pools: set[str], n: int, rng: np.random.Generator):
    """Sample n rows, at most one per group, spread evenly across splits."""
    eligible = [r for r in rows if r.pool in pools]
    if not eligible:
        return []

    one_per_group: dict[int, Any] = {}
    for r in eligible:
        one_per_group.setdefault(int(r.group_id), r)
    candidates = list(one_per_group.values())

    by_split: dict[str, list] = defaultdict(list)
    for r in candidates:
        by_split[r.split].append(r)

    picked = []
    splits = sorted(by_split)
    per = max(1, n // max(1, len(splits)))
    for s in splits:
        pool_rows = by_split[s]
        take = min(per, len(pool_rows))
        idx = rng.choice(len(pool_rows), size=take, replace=False)
        picked.extend(pool_rows[i] for i in idx)

    if len(picked) > n:
        idx = rng.choice(len(picked), size=n, replace=False)
        picked = [picked[i] for i in idx]
    return picked


def build_false_premise(rows, cfg: dict, abstain: str,
                        rng: np.random.Generator) -> list[Item]:
    out: list[Item] = []
    for name, spec in cfg["false_premise"].items():
        n = int(cfg["budget"]["false_premise"][name])
        for r in _sample_rows(rows, set(spec["pools"]), n, rng):
            options = _opts(spec["options"], abstain)
            correct = spec["correct"]
            out.append(Item(
                question_id=_qid(r.rel_path, name, spec["subtype"]),
                image_rel_path=r.rel_path, group_id=int(r.group_id),
                split=r.split, pool=r.pool, template_id=name,
                question_type=spec["subtype"], question_text=spec["text"],
                options=options, expected_behaviour=correct,
                unsafe_options=[o for o in spec["options"] if o != correct],
                mechanism_label=spec["mechanism"], answerable=True,
                provenance=f"premise contradicted by pool={r.pool}",
            ))
    return out


def build_unanswerable(rows, cfg: dict, abstain: str,
                       rng: np.random.Generator) -> list[Item]:
    out: list[Item] = []
    for name, spec in cfg["unanswerable"].items():
        n = int(cfg["budget"]["unanswerable"][name])
        for r in _sample_rows(rows, set(spec["pools"]), n, rng):
            options = _opts(spec["options"], abstain)
            out.append(Item(
                question_id=_qid(r.rel_path, name, "D"),
                image_rel_path=r.rel_path, group_id=int(r.group_id),
                split=r.split, pool=r.pool, template_id=name,
                question_type="D", question_text=spec["text"],
                options=options, expected_behaviour=abstain,
                unsafe_options=list(spec["options"]),
                mechanism_label=spec["mechanism"],
                answerable=False,          # nothing in the pixels supports an answer
                provenance="not inferable from an image",
            ))
    return out

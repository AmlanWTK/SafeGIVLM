"""Smoke tests. Run with: pytest -q"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from safegi.corruptions.ops import PILOT_SWEEP, apply
from safegi.evidence.masks import dilate, mask_stats, matched_control
from safegi.schema import Record, assign_unsafe, make_item_id, validate

H, W = 200, 240


def _frame():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.normal(150, 15, (H, W, 3)).clip(0, 255).astype("uint8"))


def _mask(box=(90, 60, 140, 110)):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return m


@pytest.mark.parametrize("name", list(PILOT_SWEEP))
def test_corruption_preserves_size_and_is_deterministic(name):
    img, msk = _frame(), _mask()
    for params in PILOT_SWEEP[name]:
        a = apply(name, img, params, lesion_mask=msk, rng=np.random.default_rng(1))
        b = apply(name, img, params, lesion_mask=msk, rng=np.random.default_rng(1))
        if a is None:
            assert b is None
            continue
        assert a.size == img.size
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{name} not deterministic"


def test_corruption_changes_the_image():
    img = _frame()
    out = apply("defocus_blur", img, {"radius": 8})
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_protected_ops_leave_the_lesion_alone():
    img, msk = _frame(), _mask()
    mk = np.asarray(msk) > 127
    orig = np.asarray(img, dtype=float)
    out = apply("debris_occlusion", img, {"coverage": 0.3, "n_patches": 4},
                lesion_mask=msk, rng=np.random.default_rng(2))
    assert np.abs(np.asarray(out, dtype=float) - orig)[mk].mean() < 1.0


def test_fov_crop_refuses_to_cut_the_lesion():
    img = _frame()
    edge = _mask(box=(5, 5, 55, 55))          # lesion at the corner
    assert apply("fov_crop", img, {"keep": 0.4}, lesion_mask=edge) is None


def test_dilate_grows_the_mask():
    mk = np.asarray(_mask()) > 127
    assert dilate(mk, 5).sum() > mk.sum()


def test_matched_control_does_not_overlap_the_lesion():
    img, msk = _frame(), _mask(box=(30, 30, 80, 80))
    cm = matched_control(img, msk)
    assert cm.ok, cm.reason
    mk = np.asarray(msk) > 127
    assert not (cm.mask & dilate(mk, int(0.05 * np.hypot(H, W)))).any()
    lesion_area = mask_stats(dilate(mk, int(0.05 * np.hypot(H, W))))["area_frac"]
    assert abs(mask_stats(cm.mask)["area_frac"] - lesion_area) < 0.01


def test_matched_control_refuses_an_oversized_lesion():
    img = _frame()
    big = _mask(box=(10, 10, 230, 190))
    assert not matched_control(img, big).ok


def _record(**over):
    base = dict(
        item_id="", image_id="kvasir-seg/x", dataset="kvasir-seg", split="est-train",
        group_id="g", question_id="q", question_type="A", question_text="Is a polyp visible?",
        options=["yes", "no"], expected_behaviour="yes", condition="clean", severity=0,
        answerable=True, mechanism_label="none", model="m", model_revision="abc1234",
        precision="fp16", prompt_id="p", answer_text="yes", chosen_option="yes",
        option_logprobs={"yes": -0.1, "no": -2.0}, gave_definite_answer=True,
        premise_accepted=None,
    )
    base.update(over)
    r = Record(**base)
    r.item_id = make_item_id(r.image_id, r.question_id, r.condition, r.severity, r.model)
    return r


def test_valid_record_passes():
    assert validate(_record()) == []


def test_unanswerable_cannot_be_correct():
    assert validate(_record(answerable=False, correct=True))


def test_moving_model_tag_is_rejected():
    assert validate(_record(model_revision="main"))


def test_unsafe_target():
    assert assign_unsafe(_record(answerable=False)) is True
    assert assign_unsafe(_record(answerable=False, gave_definite_answer=False)) is False
    assert assign_unsafe(_record(correct=False)) is True
    assert assign_unsafe(_record(correct=True)) is False


# --- question bank ---------------------------------------------------------

from safegi.questions.bank import canonical_rows, connected_components  # noqa: E402


def test_connected_components_counts_blobs():
    m = np.zeros((60, 60), dtype=bool)
    m[5:15, 5:15] = True          # 100 px
    m[40:50, 40:50] = True        # 100 px
    m[0, 59] = True               # 1 px, noise
    comps = connected_components(m, min_area=50)
    assert len(comps) == 2
    assert all(c.sum() == 100 for c in comps)


def test_connected_components_joins_an_l_shape():
    """A single region the scanline meets in two places must not split."""
    m = np.zeros((40, 40), dtype=bool)
    m[5:30, 5:10] = True
    m[25:30, 5:35] = True
    assert len(connected_components(m, min_area=10)) == 1


def test_connected_components_empty():
    assert connected_components(np.zeros((20, 20), dtype=bool), min_area=1) == []


class _Row:
    def __init__(self, group_id, dataset, pool, rel_path):
        self.group_id, self.dataset, self.pool, self.rel_path = group_id, dataset, pool, rel_path
        self.class_label = None


def test_canonical_rows_prefers_the_masked_copy():
    rows = [
        _Row(1, "hyperkvasir-labeled", "polyp", "hk/polyps/a.jpg"),
        _Row(1, "kvasir-seg-images", "polyp", "seg/images/a.jpg"),
        _Row(2, "hyperkvasir-labeled", "colonic_negative", "hk/cecum/b.jpg"),
    ]
    got = canonical_rows(rows)
    assert len(got) == 2
    assert [r.rel_path for r in got if r.group_id == 1] == ["seg/images/a.jpg"]


# --- option scoring --------------------------------------------------------

from safegi.inference.base import (  # noqa: E402
    LETTERS, build_prompt, score_item, shuffled_options,
)

_OPTS = ["yes", "no", "cannot be determined from this image"]


class _Stub:
    """Scores by option content, so reordering must not change its answer."""
    name, revision, precision = "stub", "0", "fp32"

    def __init__(self, target="no"):
        self.target = target

    def letter_logprobs(self, image, prompt, letters):
        out = {}
        for line in prompt.splitlines():
            if line[:1] in LETTERS and line[1:2] == ")":
                out[line[0]] = 2.0 if line[3:] == self.target else -1.0
        return {k: out.get(k, -9.0) for k in letters}

    def sequence_logprob(self, image, prompt, continuation):
        return -1.0, 1


class _LetterBiased(_Stub):
    """Always prefers 'A', whatever it says."""
    def letter_logprobs(self, image, prompt, letters):
        return {l: (3.0 if l == "A" else -1.0) for l in letters}


def test_option_shuffle_is_deterministic_and_total():
    a = shuffled_options("q1", _OPTS)
    assert a == shuffled_options("q1", _OPTS)
    assert sorted(a) == sorted(_OPTS)


def test_prompt_lists_every_option_once():
    prompt = build_prompt("Is a polyp visible?", _OPTS)
    for i, opt in enumerate(_OPTS):
        assert f"{LETTERS[i]}) {opt}" in prompt


def test_scores_normalise_to_one_and_pick_content():
    s = score_item(_Stub("no"), None, "q", _OPTS, "q1", with_fulltext=False)
    assert s.chosen_option == "no"
    assert abs(sum(np.exp(v) for v in s.option_logprobs.values()) - 1.0) < 1e-9
    assert set(s.option_logprobs) == set(_OPTS)


def test_answer_is_stable_under_reordering():
    seen = {score_item(_Stub("no"), None, "q", _OPTS, f"q{k}",
                       with_fulltext=False).chosen_option for k in range(6)}
    assert seen == {"no"}


def test_position_bias_is_detectable():
    """The probe's check must actually fire on a letter-biased model."""
    seen = {score_item(_LetterBiased(), None, "q", _OPTS, f"q{k}",
                       with_fulltext=False).chosen_option for k in range(6)}
    assert len(seen) > 1

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

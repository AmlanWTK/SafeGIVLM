"""
Masks for the Phase 6 evidence-grounding experiment.

The experiment compares what happens to the model's confidence when the
lesion is removed (condition B) against when an equivalent but irrelevant
region is removed (condition C). That comparison is only meaningful if C is
genuinely matched to B, so this module does the matching properly:

  - same area
  - same distance from the image centre  (polyps sit centrally; a model that
    simply weights the middle of the frame would otherwise fake the effect)
  - similar mean luminance and specular content
  - no overlap with the lesion

When no valid matched location exists — which happens for large polyps — the
generator says so instead of returning a poor control, and the caller
excludes the image and records the exclusion.

Dependencies: numpy and pillow only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = ["dilate", "ControlMask", "matched_control", "random_control", "mask_stats"]


# --- basic mask operations ---------------------------------------------------

def _as_bool(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        return np.asarray(mask.convert("L")) > 127
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127 if m.dtype != bool else m


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Binary dilation by a disc, via FFT convolution (no scipy).

    The blueprint dilates the lesion mask by 5% of the image diagonal before
    ablation, so that a thin rim of lesion does not survive at the boundary
    and leak evidence back in.
    """
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    r = int(radius)
    size = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((xx ** 2 + yy ** 2) <= r ** 2).astype(np.float32)

    ph, pw = h + 2 * r, w + 2 * r
    a = np.zeros((ph, pw), dtype=np.float32)
    a[r:r + h, r:r + w] = mask.astype(np.float32)
    k = np.zeros((ph, pw), dtype=np.float32)
    k[:size, :size] = disc
    k = np.roll(np.roll(k, -r, axis=0), -r, axis=1)

    conv = np.fft.irfft2(np.fft.rfft2(a) * np.fft.rfft2(k), s=(ph, pw))
    return conv[r:r + h, r:r + w] > 0.5


def mask_stats(mask: np.ndarray) -> dict:
    """Area fraction, centroid, and distance from centre — all normalised."""
    h, w = mask.shape
    n = int(mask.sum())
    if n == 0:
        return {"area_frac": 0.0, "cy": 0.5, "cx": 0.5, "centre_dist": 0.0, "empty": True}
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / h, xs.mean() / w
    dist = float(np.hypot(cy - 0.5, cx - 0.5))
    return {"area_frac": n / mask.size, "cy": float(cy), "cx": float(cx),
            "centre_dist": dist, "empty": False}


def _specular_frac(img_arr: np.ndarray, region: np.ndarray) -> float:
    """Fraction of a region that is near-saturated — a proxy for glare."""
    if region.sum() == 0:
        return 0.0
    v = img_arr.max(axis=2)
    return float((v[region] > 0.92).mean())


# --- matched control ---------------------------------------------------------

@dataclass
class ControlMask:
    mask: np.ndarray | None
    ok: bool
    reason: str
    luminance_delta: float = float("nan")
    specular_delta: float = float("nan")
    centre_dist_delta: float = float("nan")
    angle_deg: float = float("nan")


def matched_control(
    image: Image.Image,
    lesion: Image.Image | np.ndarray,
    *,
    dilate_frac: float = 0.05,
    max_area_frac: float = 0.40,
    n_angles: int = 72,
    min_gap_px: int = 4,
    luminance_tol: float = 0.20,
) -> ControlMask:
    """
    Build a control mask matched to the dilated lesion mask.

    The control is the lesion's own shape, rigidly translated so that its
    centroid sits on a ring about the image centre. Translation preserves
    area and shape exactly; the ring radius controls centre-distance.

    The search walks outwards: it first tries the lesion's own
    centre-distance, so the control is as close to equidistant from the
    centre as possible, and only relaxes that radius when every angle at
    the current radius overlaps the lesion or leaves the frame. This
    matters because Kvasir-SEG polyps are frequently near the centre, where
    a pure rotation about the centre would always self-overlap.

    Among valid placements at the first workable radius it picks the one
    whose mean luminance is closest to the lesion's.

    Returns ControlMask(ok=False, ...) when the polyp is too large or no
    placement works. Those images leave the experiment and the exclusion
    rate is reported.

    Note on centre_dist_delta: for a lesion sitting almost exactly at the
    image centre there is no equidistant free location, so the returned
    control is necessarily further out and centre_dist_delta will be large.
    That is reported rather than hidden. Phase 6 should either stratify on
    it or exclude above a stated threshold, and say which.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    lz = _as_bool(lesion)
    if lz.shape != (h, w):
        lz = np.asarray(
            Image.fromarray(lz.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        ) > 127

    if lz.sum() == 0:
        return ControlMask(None, False, "empty lesion mask")

    r = int(round(dilate_frac * np.hypot(h, w)))
    lesion_d = dilate(lz, r)

    stats = mask_stats(lesion_d)
    if stats["area_frac"] > max_area_frac:
        return ControlMask(None, False,
                           f"lesion too large after dilation ({stats['area_frac']:.2f} "
                           f"> {max_area_frac})")

    gap = dilate(lesion_d, min_gap_px)          # keep the control clear of the lesion
    lum = arr.mean(axis=2)
    lesion_lum = float(lum[lesion_d].mean())
    lesion_spec = _specular_frac(arr, lesion_d)

    ys, xs = np.nonzero(lesion_d)
    ly, lx = ys.mean(), xs.mean()               # lesion centroid, pixels
    cy, cx = h / 2.0, w / 2.0
    base_radius = float(np.hypot(ly - cy, lx - cx))

    # Enough separation that the translated copy can clear the lesion at all.
    extent = float(max(ys.max() - ys.min(), xs.max() - xs.min()))
    min_shift = extent + min_gap_px + 2

    # Candidate ring radii, nearest to the lesion's own first.
    radii = [base_radius]
    step = max(4.0, extent * 0.25)
    for i in range(1, 25):
        radii.append(base_radius + i * step)
        if base_radius - i * step > 0:
            radii.append(base_radius - i * step)

    best: tuple[float, np.ndarray, float, float] | None = None
    for radius in radii:
        for k in range(n_angles):
            theta = 2.0 * np.pi * k / n_angles
            ty, tx = cy + radius * np.sin(theta), cx + radius * np.cos(theta)
            if np.hypot(ty - ly, tx - lx) < min_shift:
                continue                          # too close to the lesion itself

            dy, dx = int(round(ty - ly)), int(round(tx - lx))
            ny, nx = ys + dy, xs + dx
            if ny.min() < 0 or ny.max() >= h or nx.min() < 0 or nx.max() >= w:
                continue                          # would leave the frame

            cand = np.zeros((h, w), dtype=bool)
            cand[ny, nx] = True
            if (cand & gap).any():
                continue

            delta = abs(float(lum[cand].mean()) - lesion_lum)
            if best is None or delta < best[0]:
                best = (delta, cand, np.degrees(theta), radius)
        if best is not None:
            break                                 # keep the nearest workable radius

    if best is None:
        return ControlMask(None, False, "no non-overlapping placement fits inside the frame")

    delta, cand, angle, radius = best
    cand_stats = mask_stats(cand)
    centre_delta = abs(cand_stats["centre_dist"] - stats["centre_dist"])

    if delta > luminance_tol:
        return ControlMask(cand, False,
                           f"best luminance match is poor (delta {delta:.3f} > {luminance_tol})",
                           luminance_delta=delta, centre_dist_delta=centre_delta, angle_deg=angle)

    return ControlMask(
        cand, True, "ok",
        luminance_delta=delta,
        specular_delta=abs(_specular_frac(arr, cand) - lesion_spec),
        centre_dist_delta=centre_delta,
        angle_deg=angle,
    )


def random_control(
    image: Image.Image,
    lesion: Image.Image | np.ndarray,
    *,
    dilate_frac: float = 0.05,
    min_gap_px: int = 4,
    rng: np.random.Generator | None = None,
    max_tries: int = 200,
) -> ControlMask:
    """
    Secondary control: an ellipse of matched area at a random location.

    Weaker than the rotation control because it matches area but not shape
    or centre-distance. Reported alongside it to show the result does not
    depend on the control's shape.
    """
    rng = rng or np.random.default_rng(0)
    w, h = image.size
    lz = _as_bool(lesion)
    if lz.shape != (h, w):
        lz = np.asarray(
            Image.fromarray(lz.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        ) > 127
    if lz.sum() == 0:
        return ControlMask(None, False, "empty lesion mask")

    r = int(round(dilate_frac * np.hypot(h, w)))
    lesion_d = dilate(lz, r)
    gap = dilate(lesion_d, min_gap_px)
    area = int(lesion_d.sum())
    rad = int(np.sqrt(area / np.pi))
    if rad < 2:
        return ControlMask(None, False, "lesion too small")

    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(max_tries):
        cy = int(rng.integers(rad, max(rad + 1, h - rad)))
        cx = int(rng.integers(rad, max(rad + 1, w - rad)))
        cand = ((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2
        if (cand & gap).any():
            continue
        return ControlMask(cand, True, "ok")
    return ControlMask(None, False, "no random placement avoided the lesion")

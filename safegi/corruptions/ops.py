"""
The six clinically motivated corruption operators.

Each mirrors a real way an endoscopic frame degrades: lens-to-mucosa
distance, scope movement, distant lumen, near-field reflection, residual
fluid or debris, and partial visualisation.

Two rules hold throughout.

1. Severity is a parameter here, not a clinical claim. The mapping from
   parameter to mild/moderate/severe comes from the Phase 2 endoscopist
   diagnosability pilot and lives in configs/severity_levels.yaml. Nothing
   in this file decides what "severe" means.

2. Operators that can remove evidence (overexposure, debris, crop) accept a
   lesion mask. In the answerable arm they are constrained to leave the
   lesion visible; when that is impossible the caller must label the item
   not answerable rather than pretend it is still answerable.

Dependencies: numpy and pillow only. No scipy, no cv2 — the remote
workstation and Kaggle both have these without extra installs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = [
    "defocus_blur", "motion_blur", "low_illumination", "overexposure",
    "debris_occlusion", "fov_crop", "jpeg_compress", "gaussian_noise",
    "CLINICAL_OPS", "SUPPLEMENTARY_OPS", "apply",
]


# --- helpers ----------------------------------------------------------------

def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _fft_convolve(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve HxWx3 by a 2-D kernel with wrap-free edges, via FFT."""
    h, w = arr.shape[:2]
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge")
    ph, pw = padded.shape[:2]

    k = np.zeros((ph, pw), dtype=np.float32)
    k[:kh, :kw] = kernel
    k = np.roll(k, -pad_h, axis=0)
    k = np.roll(k, -pad_w, axis=1)
    kf = np.fft.rfft2(k)

    out = np.empty_like(padded)
    for c in range(padded.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(padded[..., c]) * kf, s=(ph, pw))
    return out[pad_h:pad_h + h, pad_w:pad_w + w]


def _disc_kernel(radius: float) -> np.ndarray:
    """Circular point-spread function — the right shape for defocus."""
    r = max(1, int(round(radius)))
    size = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((xx ** 2 + yy ** 2) <= r ** 2).astype(np.float32)
    return k / k.sum()


def _line_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Linear PSF — scope or peristaltic motion during exposure."""
    n = max(3, int(round(length)) | 1)          # force odd
    k = np.zeros((n, n), dtype=np.float32)
    theta = np.deg2rad(angle_deg)
    c = n // 2
    for t in np.linspace(-c, c, n * 4):
        y = int(round(c + t * np.sin(theta)))
        x = int(round(c + t * np.cos(theta)))
        if 0 <= y < n and 0 <= x < n:
            k[y, x] = 1.0
    if k.sum() == 0:
        k[c, c] = 1.0
    return k / k.sum()


def _mask_array(mask: Image.Image | np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, Image.Image):
        m = np.asarray(mask.convert("L"))
    else:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
    if m.shape != shape:
        m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST))
    return m > 127


# --- the six clinical operators ---------------------------------------------

def defocus_blur(img: Image.Image, radius: float, **_) -> Image.Image:
    """Lens out of focal range, or fluid film on the lens. radius in pixels."""
    if radius <= 0:
        return img.copy()
    return _to_image(_fft_convolve(_to_float(img), _disc_kernel(radius)))


def motion_blur(img: Image.Image, length: float, angle: float | None = None,
                rng: np.random.Generator | None = None, **_) -> Image.Image:
    """Scope movement or peristalsis. length in pixels, angle random if unset."""
    if length <= 1:
        return img.copy()
    rng = rng or np.random.default_rng(0)
    angle = float(rng.uniform(0, 180)) if angle is None else angle
    return _to_image(_fft_convolve(_to_float(img), _line_kernel(length, angle)))


def low_illumination(img: Image.Image, gain: float, gamma: float = 1.0, **_) -> Image.Image:
    """
    Distant lumen or light-source limits. gain < 1 darkens.

    Applied in approximately linear light, so the result looks like less
    photons rather than a slider pulled down in an editor.
    """
    x = _to_float(img) ** 2.2
    x = x * float(gain)
    x = x ** (1.0 / (2.2 * gamma))
    return _to_image(x)


def overexposure(img: Image.Image, gain: float, n_highlights: int = 0,
                 highlight_radius: float = 0.06,
                 lesion_mask: Image.Image | np.ndarray | None = None,
                 protect_lesion: bool = True,
                 rng: np.random.Generator | None = None, **_) -> Image.Image:
    """
    Near-field reflection from wet mucosa: global gain plus saturated
    specular blobs.

    With protect_lesion, highlights are never placed over the lesion, which
    is what keeps the answerable arm answerable. Placing them *on* the
    lesion is a separate experiment (Phase 6), not a severity level.
    """
    rng = rng or np.random.default_rng(0)
    x = _to_float(img) ** 2.2
    x = x * float(gain)
    x = x ** (1.0 / 2.2)

    h, w = x.shape[:2]
    forbidden = _mask_array(lesion_mask, (h, w)) if protect_lesion else None
    r = max(2, int(highlight_radius * min(h, w)))

    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_highlights):
        for _attempt in range(30):
            cy = int(rng.integers(r, max(r + 1, h - r)))
            cx = int(rng.integers(r, max(r + 1, w - r)))
            if forbidden is not None:
                y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                if forbidden[y0:y1, x0:x1].any():
                    continue
            break
        else:
            continue  # nowhere safe to put it; skip this highlight
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-d2 / (2.0 * (r / 2.0) ** 2)).astype(np.float32)
        x = x + blob[..., None] * 0.9
    return _to_image(x)


def debris_occlusion(img: Image.Image, coverage: float, n_patches: int = 3,
                     lesion_mask: Image.Image | np.ndarray | None = None,
                     protect_lesion: bool = True,
                     rng: np.random.Generator | None = None, **_) -> Image.Image:
    """
    Residual stool, bubbles or blood: soft-edged opaque patches covering
    roughly `coverage` of the frame in total.

    Patches are textured and yellow-brown rather than flat grey, so the
    model is not simply detecting a rectangle.
    """
    rng = rng or np.random.default_rng(0)
    x = _to_float(img)
    h, w = x.shape[:2]
    forbidden = _mask_array(lesion_mask, (h, w)) if protect_lesion else None

    target_px = coverage * h * w
    per_patch = target_px / max(1, n_patches)
    r = int(np.sqrt(per_patch / np.pi))
    if r < 2:
        return img.copy()

    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_patches):
        for _attempt in range(40):
            cy = int(rng.integers(r, max(r + 1, h - r)))
            cx = int(rng.integers(r, max(r + 1, w - r)))
            if forbidden is not None:
                d2 = (yy - cy) ** 2 + (xx - cx) ** 2
                if forbidden[d2 <= (r * 1.15) ** 2].any():
                    continue
            break
        else:
            continue

        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        alpha = np.clip(1.2 - d2 / (r ** 2), 0.0, 1.0).astype(np.float32)
        alpha = alpha ** 0.6
        base = np.array([0.55, 0.42, 0.20], dtype=np.float32)      # bile/stool tone
        texture = rng.normal(0.0, 0.05, size=(h, w, 1)).astype(np.float32)
        patch = np.clip(base[None, None, :] + texture, 0.0, 1.0)
        x = x * (1 - alpha[..., None]) + patch * alpha[..., None]
    return _to_image(x)


def fov_crop(img: Image.Image, keep: float,
             lesion_mask: Image.Image | np.ndarray | None = None,
             require_mask_frac: float = 0.80, **_) -> Image.Image | None:
    """
    Close approach or partial visualisation: keep a centred fraction of the
    frame and resample back to the original size.

    Returns None when a lesion mask is supplied and the crop would retain
    less than require_mask_frac of it — the caller must then label the item
    not answerable instead of silently keeping a broken label.
    """
    w, h = img.size
    kw, kh = int(w * keep), int(h * keep)
    x0, y0 = (w - kw) // 2, (h - kh) // 2

    m = _mask_array(lesion_mask, (h, w))
    if m is not None and m.sum() > 0:
        kept = m[y0:y0 + kh, x0:x0 + kw].sum() / m.sum()
        if kept < require_mask_frac:
            return None
    return img.convert("RGB").crop((x0, y0, x0 + kw, y0 + kh)).resize((w, h), Image.BICUBIC)


# --- supplementary, generic robustness --------------------------------------

def jpeg_compress(img: Image.Image, quality: int, **_) -> Image.Image:
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_noise(img: Image.Image, sigma: float,
                   rng: np.random.Generator | None = None, **_) -> Image.Image:
    rng = rng or np.random.default_rng(0)
    x = _to_float(img)
    return _to_image(x + rng.normal(0.0, sigma, x.shape).astype(np.float32))


# --- registry ---------------------------------------------------------------

CLINICAL_OPS = {
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "low_illumination": low_illumination,
    "overexposure": overexposure,
    "debris_occlusion": debris_occlusion,
    "fov_crop": fov_crop,
}

SUPPLEMENTARY_OPS = {
    "jpeg": jpeg_compress,
    "gaussian_noise": gaussian_noise,
}

_ALL = {**CLINICAL_OPS, **SUPPLEMENTARY_OPS}

# Parameter sweeps for the Phase 2 diagnosability pilot. These are candidate
# levels shown to the endoscopists; the three severities are chosen FROM this
# sweep by their ratings, not by anyone's eyeball here.
PILOT_SWEEP: dict[str, list[dict]] = {
    "defocus_blur":     [{"radius": r} for r in (2, 4, 7, 11, 16)],
    "motion_blur":      [{"length": L} for L in (5, 11, 19, 29, 41)],
    "low_illumination": [{"gain": g} for g in (0.60, 0.40, 0.25, 0.15, 0.08)],
    "overexposure":     [{"gain": g, "n_highlights": n}
                         for g, n in ((1.3, 2), (1.6, 3), (2.0, 4), (2.5, 5), (3.2, 6))],
    "debris_occlusion": [{"coverage": c, "n_patches": p}
                         for c, p in ((0.05, 2), (0.12, 3), (0.22, 4), (0.35, 5), (0.50, 6))],
    "fov_crop":         [{"keep": k} for k in (0.85, 0.70, 0.55, 0.45, 0.35)],
}


def apply(name: str, img: Image.Image, params: dict,
          lesion_mask=None, rng: np.random.Generator | None = None) -> Image.Image | None:
    """Dispatch by name. Returns None only when fov_crop refuses (see above)."""
    if name == "clean":
        return img.copy()
    if name not in _ALL:
        raise KeyError(f"unknown corruption {name!r}; known: {sorted(_ALL)}")
    return _ALL[name](img, lesion_mask=lesion_mask, rng=rng, **params)

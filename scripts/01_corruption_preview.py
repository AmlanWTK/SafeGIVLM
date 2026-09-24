#!/usr/bin/env python3
"""
SafeGI-VLM  |  Phase 2, step 1  |  scripts/01_corruption_preview.py

Generates the material for the endoscopist diagnosability pilot.

For each sampled image it renders every clinical corruption at five
candidate parameter levels and writes a contact sheet plus a blinded rating
CSV. The endoscopists answer one question per panel: can the finding still
be identified? Their answers, not anyone's judgement here, decide which
three levels become mild, moderate and severe.

Panels are presented in randomised order with the level hidden, so a rater
cannot simply follow the sequence downwards. The key that maps panel ids
back to (corruption, level) is written separately and should not be opened
until the ratings are in.

Usage
-----
    python scripts/01_corruption_preview.py --root "E:/OurRe" \
        --out data/interim/pilot --n 12 --seed 7

Then: send the sheets in data/interim/pilot/sheets/ and the file
ratings_blank.csv to each rater. Keep panel_key.csv closed until both
raters return their ratings.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safegi.corruptions.ops import PILOT_SWEEP, apply  # noqa: E402

THUMB = 300
COLS = 6
PAD = 8
LABEL_H = 22


def find_pairs(root: Path, n: int, seed: int) -> list[tuple[Path, Path | None]]:
    """Sample Kvasir-SEG image/mask pairs; fall back to any images found."""
    imgs = [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and "kvasir-seg" in str(p).lower().replace("\\", "/")
            and "mask" not in str(p).lower()]
    if not imgs:
        imgs = [p for p in root.rglob("*.jpg") if "mask" not in str(p).lower()]
    if not imgs:
        sys.exit(f"No images found under {root}")

    rng = random.Random(seed)
    picked = rng.sample(imgs, min(n, len(imgs)))

    pairs = []
    for img in picked:
        mask = None
        for cand in (
            img.parent.parent / "masks" / img.name,
            img.parent.parent / "mask" / img.name,
            Path(str(img).replace("images", "masks")),
        ):
            if cand.exists() and cand != img:
                mask = cand
                break
        pairs.append((img, mask))
    return pairs


def contact_sheet(panels: list[tuple[str, Image.Image]], title: str) -> Image.Image:
    rows = (len(panels) + COLS - 1) // COLS
    w = COLS * THUMB + (COLS + 1) * PAD
    h = rows * (THUMB + LABEL_H) + (rows + 1) * PAD + 28
    sheet = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 8), title, fill=(20, 20, 20))

    for i, (label, im) in enumerate(panels):
        r, c = divmod(i, COLS)
        x = PAD + c * (THUMB + PAD)
        y = 28 + PAD + r * (THUMB + LABEL_H + PAD)
        thumb = im.copy()
        thumb.thumbnail((THUMB, THUMB), Image.LANCZOS)
        sheet.paste(thumb, (x, y))
        d.rectangle([x, y, x + thumb.width, y + thumb.height], outline=(200, 200, 200))
        d.text((x + 2, y + thumb.height + 4), label, fill=(40, 40, 40))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnosability pilot material")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/interim/pilot")
    ap.add_argument("--n", type=int, default=12, help="images to sample (blueprint uses 60)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(root, args.n, args.seed)
    print(f"sampled {len(pairs)} images "
          f"({sum(m is not None for _, m in pairs)} with masks)")

    key_rows, rating_rows = [], []
    rng = random.Random(args.seed)

    for img_path, mask_path in pairs:
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if mask_path else None
        stem = img_path.stem

        panels: list[tuple[str, Image.Image]] = []
        for corruption, sweep in PILOT_SWEEP.items():
            for level, params in enumerate(sweep):
                variant = apply(corruption, img, params, lesion_mask=mask,
                                rng=np.random.default_rng(level))
                if variant is None:
                    continue           # crop refused: it would have cut the lesion
                pid = f"{stem}_{len(panels):03d}"
                panels.append((pid, variant))
                key_rows.append({"panel_id": pid, "image": img_path.name,
                                 "corruption": corruption, "level": level,
                                 "params": str(params)})
                rating_rows.append({"panel_id": pid, "image": img_path.name,
                                    "identifiable": "", "rater": "", "notes": ""})

        rng.shuffle(panels)
        sheet = contact_sheet(
            panels,
            f"{stem} — can the finding still be identified in each panel? "
            f"(y / n / unsure)",
        )
        sheet.save(out / "sheets" / f"{stem}.jpg", quality=88)

    with (out / "panel_key.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        wtr.writeheader()
        wtr.writerows(key_rows)

    rng.shuffle(rating_rows)
    with (out / "ratings_blank.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rating_rows[0]))
        wtr.writeheader()
        wtr.writerows(rating_rows)

    print(f"\n{len(key_rows)} panels across {len(pairs)} sheets")
    print(f"  sheets        : {out / 'sheets'}")
    print(f"  rating form   : {out / 'ratings_blank.csv'}")
    print(f"  key (SEALED)  : {out / 'panel_key.csv'}")
    print("\nSend the sheets and ratings_blank.csv to each rater separately.")
    print("Do not open panel_key.csv until both sets of ratings are back.")


if __name__ == "__main__":
    main()

import cv2
import pytesseract
from typing import Dict

from app.services.kyc.pan_card_preprocessor import (
    normalize_pan_card_image
)


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh


def ocr_crop(crop) -> str:
    return pytesseract.image_to_string(
        preprocess_crop(crop),
        config="--psm 7"
    ).strip()


def ocr_pan_crop(crop) -> str:
    return pytesseract.image_to_string(
        preprocess_crop(crop),
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ).strip()


def ocr_dob_crop(crop) -> str:
    return pytesseract.image_to_string(
        preprocess_crop(crop),
        config="--psm 7 -c tessedit_char_whitelist=0123456789/"
    ).strip()


def extract_pan_fields_from_image(image_path: str) -> Dict[str, str]:
    normalized_path = normalize_pan_card_image(image_path)

    image = cv2.imread(normalized_path)

    if image is None:
        raise ValueError("Could not read normalized PAN image")

    h, w = image.shape[:2]

    crops = {
        "pan_number": (0.28, 0.30, 0.75, 0.43),
        "name": (0.08, 0.43, 0.72, 0.55),
        "father_name": (0.08, 0.55, 0.82, 0.68),
        "dob": (0.08, 0.68, 0.42, 0.82),
    }

    results = {}

    for field, (x1, y1, x2, y2) in crops.items():
        crop = image[
            int(h * y1):int(h * y2),
            int(w * x1):int(w * x2)
        ]

        if field == "pan_number":
            results[field] = ocr_pan_crop(crop)
        elif field == "dob":
            results[field] = ocr_dob_crop(crop)
        else:
            results[field] = ocr_crop(crop)

    print("PAN FIELD CROP OCR:", results)

    return results

# ---------------------------------------------------------------------------
# Date-of-birth recovery from the card image.
#
# WHY: full-page OCR reads the DOB confidently WRONG on some cards. On
# "PAN CARD OF NC MAHESH" the card prints 28/10/1958; Tesseract's full-page
# pass returned 28/10/1956 and EasyOCR returned unusable text ("20/10/105೦"),
# so the string "1958" appeared NOWHERE in either engine's output. No
# text-level reconciliation can recover a digit that neither engine ever read,
# and the wrong value was reported as "extracted" with no warning.
#
# Re-OCR'ing just the date, at higher magnification with a digits-only
# whitelist, does read it correctly. Rather than trust one hardcoded crop (the
# unused extract_pan_fields_from_image above hardcodes y 0.68-0.82, which
# misses the date entirely on these cards — layouts vary between the old
# compact card and the modern one), this scans overlapping bands down the lower
# half plus the whole card, and takes the most-voted valid date across every
# (region x scale x threshold x psm) combination.
#
# Verified on three real cards: MAHESH 28/10/1958 (13 votes vs 1),
# DEVARAJ 01/06/1969 (15 vs 1), GANESH RAO 06/10/1960 (8 vs 0).
# ---------------------------------------------------------------------------

import re as _re
from collections import Counter as _Counter

import numpy as _np
from PIL import Image as _Image

_DOB_PATTERN = _re.compile(r"\b(\d{2})[/\-.](\d{2})[/\-.](\d{4})\b")

_MIN_YEAR = 1900
_MAX_YEAR = 2100


def _is_plausible_dob(day: str, month: str, year: str) -> bool:
    return (
        1 <= int(day) <= 31
        and 1 <= int(month) <= 12
        and _MIN_YEAR <= int(year) <= _MAX_YEAR
    )


def _dob_votes(image_path: str) -> "_Counter":
    """Collect every plausible date the digit-whitelisted passes can see,
    weighted by how many independent passes agreed on it."""
    image = _np.array(_Image.open(image_path).convert("RGB"))

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    votes = _Counter()

    # Overlapping bands down the lower half (where every PAN layout puts the
    # date), plus the whole card so an unusual layout still gets a chance.
    bands = [(top, top + 0.16) for top in (0.55, 0.62, 0.69, 0.76, 0.83, 0.90)]
    regions = [(0.0, y1, 0.55, y2) for y1, y2 in bands] + [(0.0, 0.0, 1.0, 1.0)]

    for x1, y1, x2, y2 in regions:
        crop = gray[
            int(height * y1):int(height * y2),
            int(width * x1):int(width * x2)
        ]

        if crop.size == 0:
            continue

        for scale in (3, 4):
            enlarged = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

            variants = [
                enlarged,
                cv2.threshold(
                    enlarged, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1],
            ]

            for variant in variants:
                for psm in (6, 7, 11):
                    config = (
                        f"--oem 3 --psm {psm} "
                        "-c tessedit_char_whitelist=0123456789/"
                    )

                    try:
                        text = pytesseract.image_to_string(
                            variant, config=config
                        )
                    except Exception:
                        continue

                    for day, month, year in _DOB_PATTERN.findall(text):
                        if _is_plausible_dob(day, month, year):
                            votes[f"{day}/{month}/{year}"] += 1

    return votes


def recover_dob_from_image(image_path: str) -> Dict[str, object]:
    """
    Read the date of birth directly from the card image.

    Returns {"value", "votes", "runner_up", "confident"}; `value` is None when
    no plausible date was found. `confident` means the winner was seen by at
    least two passes AND beat the runner-up, which is the case where it is safe
    to override a conflicting full-page-OCR read.
    """
    empty = {"value": None, "votes": 0, "runner_up": None, "confident": False}

    try:
        votes = _dob_votes(image_path)
    except Exception as error:
        print("PAN DOB image recovery failed:", error)
        return empty

    if not votes:
        return empty

    ranked = votes.most_common(2)
    value, count = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    runner_count = ranked[1][1] if len(ranked) > 1 else 0

    return {
        "value": value,
        "votes": count,
        "runner_up": runner_up,
        "confident": count >= 2 and count > runner_count,
    }

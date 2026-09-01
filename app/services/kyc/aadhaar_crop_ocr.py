import os
import re
import cv2
import pytesseract
from typing import Dict

from app.services.kyc.aadhaar_preprocessor import (
    normalize_aadhaar_card_image
)


CROP_PROFILES = [
    {
        "profile": "standard_front_low_fields",
        "crops": {
            "name": (0.30, 0.25, 0.65, 0.34),
            "dob": (0.30, 0.45, 0.72, 0.55),
            "gender": (0.30, 0.53, 0.55, 0.63),
            "aadhaar_number": (0.32, 0.72, 0.75, 0.84),
        },
    },
    {
        "profile": "standard_front_mid_fields",
        "crops": {
            "name": (0.30, 0.27, 0.66, 0.36),
            "dob": (0.30, 0.36, 0.75, 0.48),
            "gender": (0.30, 0.46, 0.55, 0.57),
            "aadhaar_number": (0.33, 0.70, 0.78, 0.83),
        },
    },
    {
        "profile": "standard_front_high_fields",
        "crops": {
            "name": (0.32, 0.23, 0.70, 0.33),
            "dob": (0.32, 0.32, 0.78, 0.43),
            "gender": (0.32, 0.40, 0.60, 0.52),
            "aadhaar_number": (0.30, 0.68, 0.78, 0.82),
        },
    },
]


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    return thresh


def clean_value(value: str) -> str:
    value = value or ""
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def ocr_text_crop(crop) -> str:
    processed = preprocess_crop(crop)

    outputs = []

    for config in [
        "--psm 7",
        "--psm 8",
        "--psm 13",
        "--oem 3 --psm 7",
    ]:
        try:
            text = pytesseract.image_to_string(
                processed,
                config=config,
            ).strip()

            if text:
                outputs.append(text)

        except Exception:
            pass

    if not outputs:
        return ""

    outputs = sorted(
        outputs,
        key=lambda x: len(re.sub(r"[^A-Za-z]", "", x)),
        reverse=True,
    )

    return clean_value(outputs[0])


def ocr_number_crop(crop) -> str:
    text = pytesseract.image_to_string(
        preprocess_crop(crop),
        config="--psm 7 -c tessedit_char_whitelist=0123456789 ",
    )

    return clean_value(text)


def ocr_dob_crop(crop) -> str:
    text = pytesseract.image_to_string(
        preprocess_crop(crop),
        config="--psm 7 -c tessedit_char_whitelist=0123456789/-: ",
    )

    return clean_value(text)


def is_valid_aadhaar_number(value: str) -> bool:
    digits = re.sub(r"\D", "", value or "")
    return len(digits) == 12


def is_valid_dob_or_yob(value: str) -> bool:
    value = value or ""

    if re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", value):
        return True

    if re.search(r"\b(19|20)\d{2}\b", value):
        return True

    return False


def is_valid_gender(value: str) -> bool:
    upper = (value or "").upper()

    return (
        "FEMALE" in upper
        or "MALE" in upper
        or "TRANSGENDER" in upper
    )


def is_valid_name(value: str) -> bool:
    value = value or ""

    clean = re.sub(r"[^A-Za-z ]", " ", value)
    clean = re.sub(r"\s+", " ", clean).strip().upper()

    if not clean:
        return False

    bad_words = {
        "GOVERNMENT",
        "INDIA",
        "AADHAAR",
        "UIDAI",
        "UNIQUE",
        "IDENTIFICATION",
        "AUTHORITY",
        "DOB",
        "YOB",
        "YEAR",
        "BIRTH",
        "MALE",
        "FEMALE",
        "TRANSGENDER",
        "ADDRESS",
        "HUSBAND",
        "HUSSBAND",
        "FIUSSBAND",
        "FATHER",
        "MOTHER",
        "WIFE",
        "SPOUSE",
    }

    words = clean.split()

    if any(word in bad_words for word in words):
        return False

    if len(words) == 1:
        return len(words[0]) >= 4

    if len(words) > 4:
        return False

    two_letter_words = [
        word for word in words
        if len(word) == 2
    ]

    if len(two_letter_words) >= 3:
        return False

    return any(len(word) >= 4 for word in words)


def score_aadhaar_crop_result(result: Dict[str, str]) -> int:
    score = 0

    if is_valid_aadhaar_number(result.get("aadhaar_number", "")):
        score += 5

    if is_valid_dob_or_yob(result.get("dob", "")):
        score += 4

    if is_valid_gender(result.get("gender", "")):
        score += 3

    if is_valid_name(result.get("name", "")):
        score += 4

    return score


def ocr_profile(
    image,
    h: int,
    w: int,
    profile: Dict,
    debug_dir: str,
) -> Dict[str, str]:

    profile_name = profile["profile"]
    crops = profile["crops"]

    result = {
        "profile": profile_name,
        "name": "",
        "dob": "",
        "gender": "",
        "aadhaar_number": "",
    }

    profile_debug_dir = os.path.join(
        debug_dir,
        profile_name,
    )

    os.makedirs(profile_debug_dir, exist_ok=True)

    for field, (x1, y1, x2, y2) in crops.items():

        crop = image[
            int(h * y1):int(h * y2),
            int(w * x1):int(w * x2),
        ]

        crop_path = os.path.join(
            profile_debug_dir,
            f"{field}_crop.jpg",
        )

        cv2.imwrite(crop_path, crop)

        if field == "aadhaar_number":
            result[field] = ocr_number_crop(crop)

        elif field == "dob":
            result[field] = ocr_dob_crop(crop)

        else:
            result[field] = ocr_text_crop(crop)

    result["score"] = score_aadhaar_crop_result(result)

    return result


def extract_aadhaar_fields_from_image(
    image_path: str,
) -> Dict[str, str]:

    normalized_path = normalize_aadhaar_card_image(
        image_path
    )

    image = cv2.imread(normalized_path)

    if image is None:
        raise ValueError(
            "Could not read normalized Aadhaar image"
        )

    h, w = image.shape[:2]

    debug_dir = os.path.join(
        os.path.dirname(normalized_path),
        "crops",
    )

    os.makedirs(debug_dir, exist_ok=True)

    all_results = []

    for profile in CROP_PROFILES:
        profile_result = ocr_profile(
            image=image,
            h=h,
            w=w,
            profile=profile,
            debug_dir=debug_dir,
        )

        all_results.append(profile_result)

    best_result = sorted(
        all_results,
        key=lambda r: r.get("score", 0),
        reverse=True,
    )[0]

    print("AADHAAR CROP DEBUG DIR:", debug_dir)
    print("AADHAAR ALL CROP OCR:", all_results)
    print("AADHAAR BEST CROP OCR:", best_result)

    return {
        "aadhaar_number": best_result.get("aadhaar_number", ""),
        "name": best_result.get("name", ""),
        "dob": best_result.get("dob", ""),
        "gender": best_result.get("gender", ""),
        "profile": best_result.get("profile", ""),
        "score": best_result.get("score", 0),
    }
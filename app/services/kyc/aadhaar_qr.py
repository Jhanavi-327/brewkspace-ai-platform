"""
Read the QR code printed on an Aadhaar card.

WHY THIS EXISTS
---------------
OCR on a photographed Aadhaar is unreliable in ways no parsing can repair: on
one real card "S/O" came back as "52:", "Mutturamaiah" as "Mumuramaiah",
"Karnataka" as "=Kamataka", and rotated sidebar text bled fragments into the
middle of the address. The QR carries the same fields as exact machine-readable
text, so where it decodes there is nothing left to guess.

IMPORTANT — THE QR DOES NOT ALWAYS AGREE WITH THE PRINT, so it must never
silently overwrite a printed value. On a real card here the front prints
"DOB : 01/01/1952" (verified by looking at the pixels) while its QR encodes
dob="01/01/1972". Treating the QR as authoritative would have SILENTLY CHANGED
a correct extracted date to a wrong one. The caller therefore uses QR data to
fill fields OCR could not read, and reports a disagreement where both exist and
differ — which on an identity document is itself worth surfacing.

Two formats are handled:

* the older plain-XML `PrintLetterBarcodeData`, and
* the newer "Secure QR", a big decimal integer whose bytes are gzip-compressed
  and then split on 0xFF into a fixed field order.

Neither format nor field order is guessed per card — both are UIDAI's published
layouts.
"""

import re
import xml.etree.ElementTree as ET
import zlib
from typing import Dict, Optional

import cv2
import numpy as np

try:
    import zxingcpp
except Exception:  # pragma: no cover - decoder is optional
    zxingcpp = None


# Secure QR field order after gzip decompression, split on 0xFF. Index 0 is the
# email/mobile presence flag, index 1 the reference id (its first 4 digits are
# the last 4 of the Aadhaar number); the rest are the printed fields.
_SECURE_QR_FIELDS = [
    "_flags", "reference_id", "name", "dob", "gender", "care_of",
    "district", "landmark", "house", "location", "pin_code",
    "post_office", "state", "street", "sub_district", "vtc",
]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Working limits. A QR needs roughly 4 pixels per module, so magnifying a small
# crop helps and magnifying an already-large one only wastes time: upscaling a
# 4000px page render 6x produced 24000px images and made a single card take
# minutes. Large inputs are downscaled to a sane width instead.
_MAX_WORK_WIDTH = 2000
_SMALL_CROP_WIDTH = 700


def _fit_width(gray: np.ndarray, max_width: int) -> np.ndarray:
    height, width = gray.shape[:2]

    if width <= max_width:
        return gray

    scale = max_width / float(width)

    return cv2.resize(
        gray, (max_width, max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA
    )


def _decode_variants(gray: np.ndarray):
    """
    Magnify only what is too small to read, then try a global and a local
    threshold. The real card that decoded here needed 2x with an adaptive
    threshold — a raw read alone was not enough.
    """
    width = gray.shape[1]

    scales = (2, 4) if width < _SMALL_CROP_WIDTH else (1,)

    for scale in scales:
        if scale == 1:
            enlarged = gray
        else:
            enlarged = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        yield enlarged

        yield cv2.threshold(
            enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        yield cv2.adaptiveThreshold(
            enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        )


def _search_regions(height: int, width: int):
    """
    Whole image first, then overlapping halves.

    The QR sits in a different place on every layout (right of the photo on one
    card, beside the address on another), so its position is SEARCHED, not
    assumed — a hardcoded crop would work on one card and fail on the next.
    """
    yield 0, 0, width, height

    for top in (0.0, 0.25, 0.5):
        for left in (0.0, 0.25, 0.5):
            yield (
                int(width * left), int(height * top),
                int(width * (left + 0.5)), int(height * (top + 0.5)),
            )


def decode_qr_from_image(image: np.ndarray) -> Optional[str]:
    """Return the QR payload found in `image`, or None."""
    if zxingcpp is None:
        return None

    gray = _fit_width(_to_gray(image), _MAX_WORK_WIDTH)
    height, width = gray.shape[:2]

    for x1, y1, x2, y2 in _search_regions(height, width):
        crop = gray[y1:y2, x1:x2]

        if crop.size == 0 or min(crop.shape[:2]) < 60:
            continue

        for variant in _decode_variants(crop):
            try:
                results = zxingcpp.read_barcodes(variant)
            except Exception:
                continue

            for result in results:
                if result.text:
                    return result.text

    return None


def _parse_xml_payload(payload: str) -> Optional[Dict[str, Optional[str]]]:
    """Older QR format: a PrintLetterBarcodeData XML element."""
    if "PrintLetterBarcodeData" not in payload:
        return None

    try:
        node = ET.fromstring(payload.strip())
    except Exception:
        return None

    get = node.attrib.get

    return {
        "aadhaar_number": get("uid"),
        "name": get("name"),
        "gender": get("gender"),
        "date_of_birth_or_year": get("dob") or get("yob"),
        "guardian_name": get("gname") or get("co"),
        "house": get("house"),
        "street": get("street"),
        "landmark": get("lm"),
        "location": get("loc"),
        "vtc": get("vtc"),
        "post_office": get("po"),
        "sub_district": get("subdist"),
        "district": get("dist"),
        "state": get("state"),
        "pin_code": get("pc"),
    }


def _parse_secure_qr(payload: str) -> Optional[Dict[str, Optional[str]]]:
    """Newer format: a big decimal integer, gzip-compressed, 0xFF separated."""
    digits = payload.strip()

    if not digits.isdigit() or len(digits) < 40:
        return None

    try:
        value = int(digits)
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        decompressed = zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    except Exception:
        return None

    parts = decompressed.split(b"\xff")

    if len(parts) < len(_SECURE_QR_FIELDS):
        return None

    values = {}

    for index, field in enumerate(_SECURE_QR_FIELDS):
        try:
            values[field] = parts[index].decode("utf-8", "ignore").strip()
        except Exception:
            values[field] = ""

    return {
        # The Secure QR deliberately carries only the last 4 Aadhaar digits.
        "aadhaar_number": None,
        "name": values.get("name") or None,
        "gender": values.get("gender") or None,
        "date_of_birth_or_year": values.get("dob") or None,
        "guardian_name": values.get("care_of") or None,
        "house": values.get("house") or None,
        "street": values.get("street") or None,
        "landmark": values.get("landmark") or None,
        "location": values.get("location") or None,
        "vtc": values.get("vtc") or None,
        "post_office": values.get("post_office") or None,
        "sub_district": values.get("sub_district") or None,
        "district": values.get("district") or None,
        "state": values.get("state") or None,
        "pin_code": values.get("pin_code") or None,
    }


def parse_qr_payload(payload: str) -> Optional[Dict[str, Optional[str]]]:
    if not payload:
        return None

    return _parse_xml_payload(payload) or _parse_secure_qr(payload)


def _normalise_gender(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    initial = value.strip().upper()[:1]

    return {"M": "MALE", "F": "FEMALE", "T": "TRANSGENDER"}.get(initial)


def _normalise_number(value: Optional[str]) -> Optional[str]:
    digits = re.sub(r"\D", "", value or "")

    if len(digits) != 12:
        return None

    return f"{digits[:4]} {digits[4:8]} {digits[8:]}"


def build_qr_address(data: Dict[str, Optional[str]]) -> Optional[str]:
    """Join the QR's address components in the order UIDAI prints them."""
    order = (
        "house", "street", "landmark", "location",
        "vtc", "post_office", "sub_district", "district",
        "state", "pin_code",
    )

    seen = []

    for key in order:
        value = (data.get(key) or "").strip(" ,")

        # The same place often fills two components (vtc and post office, say).
        if value and value.lower() not in [s.lower() for s in seen]:
            seen.append(value)

    return ", ".join(seen) or None


def read_aadhaar_qr(images) -> Optional[Dict[str, Optional[str]]]:
    """
    Decode the first readable Aadhaar QR among `images`.

    Returns the parsed fields (normalised gender / Aadhaar number, plus a
    joined `address`) and the raw payload, or None when nothing decodes —
    which is common, since a Secure QR on a low-resolution phone photo of a
    card often carries too few pixels per module to read.
    """
    for image in images:
        payload = decode_qr_from_image(image)

        if not payload:
            continue

        parsed = parse_qr_payload(payload)

        if not parsed:
            continue

        parsed["gender"] = _normalise_gender(parsed.get("gender"))
        parsed["aadhaar_number"] = _normalise_number(parsed.get("aadhaar_number"))
        parsed["address"] = build_qr_address(parsed)
        parsed["raw_payload"] = payload

        return parsed

    return None

import json
import re
import time
from typing import Dict, Any, Optional

import fitz
from PIL import Image
from google import genai

from app.core.config import settings


AADHAAR_REGEX = r"\b\d{4}[\s_]*\d{4}[\s_]*\d{4}\b"


# Tried in order. The free tier enforces a per-model daily request quota,
# so when the primary model is exhausted (429 RESOURCE_EXHAUSTED) the next
# model has its own separate quota bucket. This keeps the accurate vision
# path alive far longer before falling back to crop / OCR extraction.
# VERIFIED CALLABLE on this project key 2026-08-31. The previous list was
# entirely / partly RETIRED — Google returns 404 "no longer available to new
# users" for every gemini-2.5-* and gemini-2.0-* name, so those entries could
# never answer and each burned 3 retries before the next was tried. Google's own
# migration targets: 2.5-flash and 2.0-flash -> 3.6-flash, 2.5-flash-lite and
# 2.0-flash-lite -> 3.5-flash-lite. Do NOT add a name without confirming it
# answers on this key; models.list() also returns names that 404.
AADHAAR_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def _is_quota_error(error) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


AADHAAR_PROMPT = """
You are extracting data from an Indian Aadhaar card image or PDF.

Return ONLY valid JSON. No markdown. No explanation.

Extract exactly these fields:

{
  "document_type": "AADHAAR_CARD",
  "aadhaar_number": null,
  "name": null,
  "date_of_birth_or_year": null,
  "gender": null,
  "address": null,
  "pin_code": null
}

Rules:
- "aadhaar_number" must be 12 digits formatted as XXXX XXXX XXXX.
- "name" is the Aadhaar holder's own name ONLY. It is printed near the
  photo, above the date of birth. Do NOT return the father's, husband's,
  mother's or guardian's name (labelled S/O, D/O, W/O, C/O, Father,
  Husband). If only that relation name is visible, still return the
  holder's name if you can read it, otherwise null.
- The name may be printed in English and in a regional script
  (Kannada, Hindi, etc.). Return the ENGLISH spelling in uppercase.
- "date_of_birth_or_year" can be DD/MM/YYYY or a 4 digit year only.
- "gender" must be MALE, FEMALE, or TRANSGENDER.
- "address" is optional and usually on the back side.
- If a field is not visible, use null. Do not guess.
"""


def clean_json_response(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

    return {}


def normalize_aadhaar_number(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    digits = re.sub(r"\D", "", value)

    if len(digits) != 12:
        return None

    return f"{digits[:4]} {digits[4:8]} {digits[8:]}"


def normalize_gender(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = value.upper()

    if "FEMALE" in value:
        return "FEMALE"

    if "MALE" in value:
        return "MALE"

    if "TRANSGENDER" in value:
        return "TRANSGENDER"

    return None


def normalize_dob(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    dob = re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", value)

    if dob:
        return dob.group().replace("-", "/")

    yob = re.search(r"\b\d{4}\b", value)

    if yob:
        return yob.group()

    return None


def normalize_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = re.sub(r"[^A-Za-z ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip().upper()

    if len(value) < 4:
        return None

    return value


def _pdf_first_page_to_image(pdf_path: str, dpi: int = 300) -> Image.Image:
    pdf = fitz.open(pdf_path)
    page = pdf[0]

    pix = page.get_pixmap(dpi=dpi)

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    pdf.close()

    return image


def _load_image(file_path: str) -> Image.Image:
    if file_path.lower().endswith(".pdf"):
        return _pdf_first_page_to_image(file_path)
    return Image.open(file_path).convert("RGB")


def extract_aadhaar_card_gemini(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract Aadhaar card fields directly from the image using Gemini vision.

    Returns a dict matching the schema produced by the regex extractor,
    or None if Gemini is unavailable or extraction fails so the caller
    can fall back to the OCR + regex parser.
    """
    if not settings.GOOGLE_API_KEY:
        return None

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    image = _load_image(file_path)

    # Try each model in turn. Within a model, retry transient failures with
    # exponential backoff. A daily-quota (429) error is NOT transient for
    # that model, so we stop retrying it and move straight to the next model,
    # which has its own separate quota bucket.
    response = None
    last_error = None
    used_model = None

    for model in AADHAAR_MODELS:
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[AADHAAR_PROMPT, image]
                )
                if response and getattr(response, "text", None):
                    break
            except Exception as e:
                last_error = e
                print(f"Gemini Aadhaar {model} attempt {attempt + 1} failed:", e)

                # Quota exhaustion won't clear on retry - fall through to the
                # next model immediately instead of burning backoff time.
                if _is_quota_error(e):
                    break

                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)

        if response is not None and getattr(response, "text", None):
            used_model = model
            print(f"Gemini Aadhaar extracted using model: {model}")
            break

    if response is None or not getattr(response, "text", None):
        if last_error:
            print("Gemini Aadhaar extraction giving up after all models:", last_error)
        return None

    data = clean_json_response(response.text)

    if not isinstance(data, dict):
        return None

    aadhaar_number = normalize_aadhaar_number(data.get("aadhaar_number"))
    name = normalize_name(data.get("name"))
    dob = normalize_dob(data.get("date_of_birth_or_year"))
    gender = normalize_gender(data.get("gender"))

    address = data.get("address")
    pin_code = data.get("pin_code")

    if address:
        address = re.sub(r"\s+", " ", str(address)).strip()

    if pin_code:
        pin_digits = re.sub(r"\D", "", str(pin_code))
        pin_code = pin_digits if len(pin_digits) == 6 else None

    field_status = {
        "aadhaar_number": "extracted" if aadhaar_number else "requires_manual_review",
        "name": "extracted" if name else "requires_manual_review",
        "date_of_birth_or_year": "extracted" if dob else "requires_manual_review",
        "gender": "extracted" if gender else "requires_manual_review",
        "address": "extracted" if address else "requires_manual_review",
        "pin_code": "extracted" if pin_code else "requires_manual_review",
    }

    return {
        "document_type": "AADHAAR_CARD",
        "aadhaar_number": aadhaar_number,
        "name": name,
        "date_of_birth_or_year": dob,
        "gender": gender,
        "address": address,
        "pin_code": pin_code,
        "field_status": field_status,
        "extraction_status": (
            "extracted"
            if aadhaar_number and name and dob and gender
            else "partial_requires_review"
        ),
        "extraction_source": f"aadhaar_gemini_vision_v1:{used_model}",
    }

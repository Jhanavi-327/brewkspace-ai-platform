import json
import re
import time

import fitz
from PIL import Image
from google import genai

from app.core.config import settings


PAN_PROMPT = """
You are extracting fields from an Indian Income Tax Department
Permanent Account Number (PAN) card image.

Return ONLY valid JSON. No markdown. No explanation.

Extract exactly these fields:

{
  "pan_number": null,
  "name": null,
  "father_name": null,
  "date_of_birth": null
}

Rules:
- "pan_number" is the 10 character code in the format AAAAA9999A
  (5 letters, 4 digits, 1 letter). Read it carefully and do not
  confuse letters and digits (O vs 0, I vs 1, S vs 5, B vs 8, Z vs 2).
- "name" is the card holder's name printed under the label "Name".
- "father_name" is printed under the label "Father's Name".
- "date_of_birth" is printed under "Date of Birth". Always return it
  in zero-padded DD/MM/YYYY format (two digits, two digits, four digits).
  Report only the digits printed on this card; never complete or adjust a
  date towards any other value.
- Use uppercase for names exactly as printed.
- If a field is not visible, use null. Do not guess.
"""


PAN_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"


# Tried in order. The free tier enforces a per-model daily request quota,
# so when the primary model is exhausted (429 RESOURCE_EXHAUSTED) the next
# model has its own separate quota bucket. flash is the most accurate on
# unusual card layouts; flash-lite is the availability backup.
# VERIFIED CALLABLE on this project key 2026-08-31. The previous list was
# entirely / partly RETIRED — Google returns 404 "no longer available to new
# users" for every gemini-2.5-* and gemini-2.0-* name, so those entries could
# never answer and each burned 3 retries before the next was tried. Google's own
# migration targets: 2.5-flash and 2.0-flash -> 3.6-flash, 2.5-flash-lite and
# 2.0-flash-lite -> 3.5-flash-lite. Do NOT add a name without confirming it
# answers on this key; models.list() also returns names that 404.
PAN_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def _is_quota_error(error) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _extract_json(text: str):
    if not text:
        return {}

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}


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


def _normalize_dob(dob):
    if not dob:
        return None

    dob = str(dob).strip()
    dob = dob.replace("-", "/").replace(".", "/")

    parts = dob.split("/")
    if len(parts) != 3:
        return None

    day, month, year = parts

    if not (day.isdigit() and month.isdigit() and year.isdigit()):
        return None

    if len(year) == 2:
        year = ("19" if int(year) > 30 else "20") + year

    if not (1 <= int(day) <= 31):
        return None
    if not (1 <= int(month) <= 12):
        return None
    if len(year) != 4:
        return None

    return f"{day.zfill(2)}/{month.zfill(2)}/{year}"


def _normalize_pan(pan):
    if not pan:
        return None

    pan = re.sub(r"[^A-Za-z0-9]", "", str(pan)).upper()

    if re.fullmatch(PAN_REGEX, pan):
        return pan

    return None


def _normalize_name(name):
    if not name:
        return None

    name = re.sub(r"\s+", " ", str(name)).strip().upper()

    return name or None


def extract_pan_card_gemini(file_path: str):
    """
    Extract PAN card fields directly from the image using Gemini vision.

    Returns a dict matching the schema produced by the regex extractor,
    or None if Gemini is unavailable or extraction fails so the caller
    can fall back to the OCR + regex parser.
    """
    if not settings.GOOGLE_API_KEY:
        return None

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    image = _load_image(file_path)

    # Try each model in turn. Within a model, retry transient failures with
    # exponential backoff so a single flaky call does not drop the document
    # to the much less reliable OCR + regex fallback. A daily-quota (429)
    # error is NOT transient for that model, so we stop retrying it and move
    # straight to the next model, which has its own separate quota bucket.
    response = None
    last_error = None
    used_model = None

    for model in PAN_MODELS:
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[PAN_PROMPT, image]
                )
                if response and getattr(response, "text", None):
                    break
            except Exception as e:
                last_error = e
                print(f"Gemini PAN {model} attempt {attempt + 1} failed:", e)

                # Quota exhaustion won't clear on retry - fall through to the
                # next model immediately instead of burning backoff time.
                if _is_quota_error(e):
                    break

                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)

        if response is not None and getattr(response, "text", None):
            used_model = model
            print(f"Gemini PAN extracted using model: {model}")
            break

    if response is None or not getattr(response, "text", None):
        if last_error:
            print("Gemini PAN extraction giving up after all models:", last_error)
        return None

    data = _extract_json(response.text)

    if not isinstance(data, dict):
        return None

    pan_number = _normalize_pan(data.get("pan_number"))
    name = _normalize_name(data.get("name"))
    father_name = _normalize_name(data.get("father_name"))
    date_of_birth = _normalize_dob(data.get("date_of_birth"))

    return {
        "document_type": "PAN_CARD",
        "pan_number": pan_number,
        "possible_pan_tokens": [pan_number] if pan_number else [],
        "name": name,
        "father_name": father_name,
        "date_of_birth": date_of_birth,
        "category": "INDIVIDUAL",
        "field_status": {
            "pan_number": "extracted" if pan_number else "requires_manual_review",
            "name": "extracted" if name else "requires_manual_review",
            "father_name": "extracted" if father_name else "requires_manual_review",
            "date_of_birth": "extracted" if date_of_birth else "requires_manual_review",
        },
        "extraction_status": (
            "extracted"
            if pan_number and name and father_name and date_of_birth
            else "partial_requires_review"
        ),
        "extraction_source": f"gemini_vision_pan_v1:{used_model}",
    }

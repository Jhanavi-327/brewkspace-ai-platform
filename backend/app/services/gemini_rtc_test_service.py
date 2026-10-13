import json
import re
import fitz

from PIL import Image
from google import genai

from app.core.config import settings


RTC_TEST_PROMPT = """
You are extracting ONLY the complex table fields from a Karnataka RTC / Pahani land record.

Return ONLY valid JSON. No markdown. No explanation.

Do not extract taluk, hobli, village, survey number, hissa, or total extent.

Extract only these fields:

{
  "owners": [
    {
      "name": null,
      "extent": null,
      "khata_no": null,
      "possession_type": null
    }
  ],
  "rights_liabilities": {
    "mr_number": null,
    "mutation_date": null,
    "court_reference": null,
    "rights_text": null,
    "liabilities_text": null
  },
  "cultivation_details": [
    {
      "year": null,
      "season": null,
      "cultivator_name": null,
      "cultivation_method": null,
      "land_use": null,
      "crop_name": null,
      "wet_extent": null,
      "dry_extent": null,
      "total_extent": null,
      "irrigation_source": null,
      "liability_amount": null,
      "mixed_crop_name": null,
      "mixed_crop_extent": null
    }
  ],
  "other_rights": {
    "restrictions": null,
    "remarks": null
  }
}

Rules:
- Preserve Kannada text exactly in Kannada Unicode.
- Do not guess.
- Do not copy column headings as values.
- If a field is not visible or not present, use null.
- If the cultivator is the same as owner, still write the visible cultivator name.
- If multiple seasons are visible, create one cultivation_details item per season.
"""


def _extract_json(text: str):
    if not text:
        return {"raw_response": text}

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

    return {"raw_response": text}


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


def rtc_gemini_test(file_path: str):
    if not settings.GOOGLE_API_KEY:
        return {
            "error": "GOOGLE_API_KEY is missing in .env"
        }

    client = genai.Client(
        api_key=settings.GOOGLE_API_KEY
    )

    if file_path.lower().endswith(".pdf"):
        image = _pdf_first_page_to_image(file_path)
    else:
        image = Image.open(file_path).convert("RGB")

    response = client.models.generate_content(
        # gemini-2.5-flash is retired on this key (404). Superseded by 3.6-flash.
        model="gemini-3.6-flash",
        contents=[
            RTC_TEST_PROMPT,
            image
        ]
    )

    return _extract_json(response.text)
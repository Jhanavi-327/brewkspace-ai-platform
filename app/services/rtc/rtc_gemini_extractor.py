"""
Karnataka RTC / Pahani (ರೆಕಾರ್ಡ್ ಆಫ್ ರೈಟ್ಸ್, ಟೆನೆನ್ಸಿ ಮತ್ತು ಪಹಣಿ ಪತ್ರಿಕೆ — Village
Account Form No. 2) extraction with Gemini vision.

WHY THIS EXISTS
---------------
RTC was the last document type still extracted by Tesseract cell-OCR
(rtc_table_service -> rtc_cell_ocr_service -> rtc_cell_block_mapper). On real
landrecords.karnataka.gov.in PDFs that pipeline fails in two ways that no
amount of downstream regex mapping can repair:

  1. Table detection collapses the 13-block form into a handful of merged
     mega-cells. On "RTC 12-3.pdf" a single cell spanned y1349-1972 and held
     the whole ownership column AND both cultivation rows as one blob.
  2. Kannada recognition substitutes Latin glyphs — "ಕೋಂ" (w/o, the standard
     RTC relationship marker) came back as "BOLO" and "BWOlO" in that cell.

Because the blocks are merged, the mapper also copies PRINTED FORM LABELS as
values: the "5." of the section heading "5. ಮಣ್ಣಿನ ನಮೂನೆ" was emitted as
hissa_number, and the pre-printed words ಕಪ್ಪು / ಸರ್ಕಾರಿ were emitted as
soil_type / land_classification even though those boxes are blank on the page.

This module reads the rendered page image directly, mirroring the PAN /
Aadhaar / Patta / MR extractors: the same multi-model quota fallback, the same
OCR-text grounding, and — because an RTC carries no printed arithmetic total to
checksum against — the MR-style second-model text-agreement pass as the trust
signal instead. Unlike those extractors it distinguishes a per-model daily
quota (fall through to the next name) from a project-level spend cap (shared by
every model — abort the chain), and its model list has been verified callable
rather than inherited.

The returned dict keeps the EXACT block keys the existing pipeline contract
uses (header, 1_survey_number ... 13_other_rights), so documents.py, the
property-record save and the frontend need no shape change.
"""

import json
import re
import time

import fitz
from google import genai
from google.genai import types
from PIL import Image

from app.core.config import settings


GENERATION_CONFIG = types.GenerateContentConfig(temperature=0)


# Tried in order, accuracy-first. Falling through models helps ONLY for a
# per-model daily quota, where the next name draws on its own bucket. It does
# NOT help for a project-level spend cap, which every model on the key shares —
# see _is_spend_cap_error, which aborts the chain instead of burning 3 retries
# per model against a cap that is already exhausted.
#
# VERIFIED CALLABLE on this project key 2026-08-31 by an actual image request.
# Do NOT add a name without that check: models.list() returns plenty that 404
# here, and gemini-2.5-flash / gemini-2.5-flash-lite (in the original list,
# copied from the Patta / MR extractors) are BOTH retired on this key —
# "no longer available to new users", superseded by the 3.5 line. Their
# presence meant the chain had no real depth once the top two were exhausted.
RTC_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


RTC_PROMPT = """
You are extracting fields from a Karnataka RTC / Pahani land record
(ರೆಕಾರ್ಡ್ ಆಫ್ ರೈಟ್ಸ್, ಟೆನೆನ್ಸಿ ಮತ್ತು ಪಹಣಿ ಪತ್ರಿಕೆ — Village Account Form No. 2)
issued from landrecords.karnataka.gov.in.

Return ONLY valid JSON. No markdown. No explanation.

THE FORM
--------
The page is one large printed table. Every section is PRE-PRINTED with a
numbered heading (for example "1. ಸರ್ವೆ ನಂಬರ", "2. ಹಿಸ್ಸಾ", "5. ಮಣ್ಣಿನ ನಮೂನೆ",
"9. ಸ್ವಾಧೀನದಾರರ ಹೆಸರು"). The VALUE is the entry filled into that section's box
— it is NOT the heading itself.

Many sections on a given RTC are legitimately BLANK. A blank section must come
back as null. Returning a plausible-looking value for a blank box is the single
worst failure mode on this document — worse than returning null.

Extract exactly this structure:

{
  "header": {
    "taluk": null,
    "hobli": null,
    "village": null,
    "valid_from": null,
    "valid_to": null
  },
  "1_survey_number": { "survey_number": null },
  "2_hissa": { "hissa_number": null },
  "3_khata_vara": {
    "total_extent": null,
    "phut_kharab_a": null,
    "phut_kharab_b": null,
    "remaining_extent": null
  },
  "4_land_revenue": {
    "land_revenue": null,
    "jodi": null,
    "cess": null,
    "water_rate": null,
    "total_revenue": null
  },
  "5_soil_type": { "soil_type": null },
  "6_land_classification": { "land_classification": null },
  "7_tree_details": { "tree_count": null, "tree_types": [] },
  "8_irrigation_details": {
    "source": null,
    "kharif_extent": null,
    "rabi_extent": null,
    "garden_extent": null,
    "total_extent": null
  },
  "9_ownership_details": [
    {
      "owner_name": null,
      "father_or_spouse_name": null,
      "address": null,
      "extent": null,
      "khata_no": null,
      "possession_type": null
    }
  ],
  "10_possession_details": {
    "possession_type": null,
    "owner_wise_possession": [
      { "owner_name": null, "possession_type": null }
    ]
  },
  "11_rights_liabilities": {
    "mr_number": null,
    "court_stay": false,
    "court_reference": null,
    "recorded_on": null,
    "rights": null,
    "liabilities": null,
    "remarks": null
  },
  "12_cultivation_details": [
    {
      "crop_year": null,
      "season": null,
      "cultivator_name": null,
      "cultivation_method": null,
      "land_use": null,
      "crop_name": null,
      "crop_extent": null,
      "irrigated_extent": null
    }
  ],
  "13_other_rights": {
    "other_rights": null,
    "restrictions": null,
    "remarks": null
  }
}

SECTION-BY-SECTION RULES
------------------------
header: taluk follows "ತಾಲ್ಲೂಕು:", hobli follows "ಹೋಬಳಿ:", village follows
  "ಗ್ರಾಮ:". valid_from / valid_to come from the "Valid from <date> To <date>"
  line at the top right; keep whatever form is printed — a DD/MM/YYYY date, or
  wording such as "Till Date" where no end date is given.

1_survey_number / 2_hissa: two SEPARATE small boxes in the left column. The
  survey number sits under the heading "1. ಸರ್ವೆ ನಂಬರ" and the hissa number
  under "2. ಹಿಸ್ಸಾ". Read ONLY the digits entered in the box. A leading section
  number from the NEXT heading (the "5." of "5. ಮಣ್ಣಿನ ನಮೂನೆ", the "6." of
  "6. ಪಟ್ಟಾ") is NOT the hissa value. If the hissa box is empty, return null.

3_khata_vara: the extent column, written as four dot-separated numeric groups
  (ACRE.GUNTA.ANNA.FRACTION). Copy the digits exactly as printed and do not
  reformat or pad them. total_extent is the ಒಟ್ಟು ವಿಸ್ತೀರ್ಣ row, phut_kharab_a the
  ಪೂಟ್ ಖರಾಬ್ (ಅ) row, phut_kharab_b the ಪೂಟ್ ಖರಾಬ್ (ಬ) row, remaining_extent
  the ಉಳಿದದ್ದು row.

4_land_revenue: the ರೂ./ಪೈ money column under "4. ಕಂದಾಯ" —
  (ಅ) ಭೂ ಕಂದಾಯ = land_revenue, (ಬ) ಜೋಡಿ = jodi, (ಕ) ಸೆಸ್ಸುಗಳು = cess,
  (ಡ) ನೀರಿನ ದರ = water_rate, ಒಟ್ಟು = total_revenue.

5_soil_type and 6_land_classification: each is a heading with the entry
  printed BELOW it in the same narrow left-hand box. Both are short
  single-word categories — a soil class and a tenure class respectively. Read
  whichever word is actually printed; do NOT pick from a list of expected
  categories, and do not normalise the wording. These boxes are
  usually FILLED — read the black entry under the blue heading. Return null
  only when the space under the heading is genuinely empty.

7_tree_details: "7. ಮರಗಳ ಸಂಖ್ಯೆ" has ಹೆಸರು (name) and ಸಂಖ್ಯೆ (count)
  sub-columns. tree_count must be a count written in the ಸಂಖ್ಯೆ sub-column.
  The digit "8" of the neighbouring heading "8. ಬೇಸಾಯ ಪ್ರಕಾರ ನೀರಾವರಿಯ
  ವಿಸ್ತೀರ್ಣ" is a SECTION NUMBER, never a tree count.

8_irrigation_details: from the ನೀರಾವರಿ ಮೂಲ / ಮುಂಗಾರು / ಹಿಂಗಾರು / ಬಾಗಾಯ್ತು /
  ಒಟ್ಟು sub-columns. These are usually blank — return nulls when blank.

9_ownership_details: column 9 "ಸ್ವಾಧೀನದಾರರ ಹೆಸರು ಮತ್ತು ವಿಳಾಸ". One array item
  per DISTINCT owner. Karnataka names use relationship markers — ಬಿನ್ (son of),
  ಕೋಂ (wife of), ಮಗ, ಮಗಳು, ಲೇ (late). Put the person's own name in owner_name
  and the parent/spouse named after the marker in father_or_spouse_name.
  extent and khata_no come from the ವಿಸ್ತೀರ್ಣ (ಎ/ಗುಂ) and ಖಾತೆ ನಂ. sub-columns
  beside the name. Do not repeat the same person as two items, and do not merge
  two different people into one string.
  SHORT-FORM THEN EXPANDED FORM IS ONE PERSON. This column routinely prints a
  holder's bare name on one line and then the SAME name again below it in full,
  with the relationship marker and the parent/spouse — the pattern is
  "<HOLDER>" on one line, then "<HOLDER> ಕೋಂ <SPOUSE>" (or ಬಿನ್ / ಮಗ, and
  optionally ಲೇ before the parent/spouse) below it. That is ONE
  owner, not two: emit a single item, using the fuller form for owner_name and
  the person after the marker for father_or_spouse_name. Spelling may differ
  slightly between the two lines (a different sibilant, or a doubled vs single
  consonant / vowel length) — that is the same name, not a second holder. A genuinely separate co-owner has its OWN extent
  or khata number, or is joined by ಮತ್ತು / ಹಾಗೂ.

10_possession_details: column 10 "ಕಬ್ಜೆ ಅಥವಾ ಸ್ವಾಧೀನತೆಯ ರೀತಿ" — the manner of
  holding, e.g. ಸ್ವಂತ (own), ಜಂಟಿ (joint), ಪೌತಿ (inheritance), ಗುತ್ತಿಗೆ
  (lease). possession_type is the list of distinct kinds visible in that
  column. owner_wise_possession pairs each owner_name from section 9 with the
  kind recorded on that owner's row. Text belonging to a neighbouring column
  that merely overlaps this one visually is NOT a possession type.

11_rights_liabilities: column 11 "ಇತರೆ ಹಕ್ಕುಗಳು ಮತ್ತು ಋಣಗಳು". mr_number is the
  mutation reference, printed as "MR" followed by a number and a year range.
  Copy it exactly as printed. recorded_on is the date printed
  with it. Set court_stay true only if a stay / ತಡೆಯಾಜ್ಞೆ is actually recorded,
  and put its case reference in court_reference.

12_cultivation_details: the wide bottom table (columns 1-16). Emit ONE item per
  visible DATA ROW. crop_year is the ವರ್ಷ column (a YYYY-YYYY range), season the
  ಕಾಲ column (ಮುಂಗಾರು = kharif, ಹಿಂಗಾರು = rabi, ಬೇಸಿಗೆ = summer),
  cultivator_name the ಸಾಗುವಳಿದಾರರ ಹೆಸರು column, cultivation_method the ಸಾಗುವಳಿ
  ವಿಧ column (ಸ್ವಂತ / ಗುತ್ತಿಗೆ), land_use the ಭೂಮಿಯ ಉಪಯೋಗ column, crop_name the
  ಬೆಳೆಯ ಹೆಸರು column, crop_extent and irrigated_extent the matching ವಿಸ್ತೀರ್ಣ
  columns. If the bottom table has no data rows, return [].
  THE SHORT-FORM/EXPANDED-FORM MERGE RULE FROM SECTION 9 DOES NOT APPLY
  HERE. This table is a per-row register: emit one item for EVERY printed
  data row, even when two rows name the same cultivator, and never drop a
  row because its name resembles another row's. Each season block repeats
  the full set of cultivator rows, so the same names recur under ಮುಂಗಾರು
  and again under ಹಿಂಗಾರು — that is expected, not duplication. Copy each
  row's cultivator name exactly as printed on that row. In particular, a
  row naming a DIFFERENT person (for example a late spouse who appears in
  section 9 only as a relationship) is its own row and must be kept.

13_other_rights: any ಇತರೆ ಹಕ್ಕುಗಳು / restriction / remark text not already
  captured above.

GENERAL RULES
-------------
- COLOUR TELLS HEADING FROM ENTRY. The blank form is pre-printed in BLUE: every
  section heading, column heading and section number is blue. The values filled
  in for this particular parcel are printed in BLACK. When deciding whether
  something is a heading or a value, use the colour first — a black string
  under a blue heading is that section's value, and a blue string is never a
  value no matter how plausible it looks.
- Preserve Kannada text exactly in Kannada Unicode. Never transliterate Kannada
  into Latin letters. If a Kannada word is unclear, return null rather than a
  Latin approximation — "BOLO" or "BWOlO" in place of "ಕೋಂ" is always wrong.
- Never copy a pre-printed section heading, column heading or its number as a
  value.
- Never invent, average or compute a value. Read only what is printed.
- Keep numbers in their printed form; do not reformat or pad extents.
- Ignore the "First / Previous / Next / Last" navigation buttons, the browser
  URL and timestamp, the "For Viewing Only" watermark, the "RTC DIGITALLY
  SIGNED ON" footer and the barcode.
- If a whole section is blank, return that section with null fields (and [] for
  9_ownership_details / 12_cultivation_details) rather than omitting it.
"""


# Blocks that must always be present in the returned dict, with the shape the
# downstream pipeline expects when a model omits them entirely.
_EMPTY_BLOCKS = {
    "header": {
        "taluk": None,
        "hobli": None,
        "village": None,
        "valid_from": None,
        "valid_to": None,
    },
    "1_survey_number": {"survey_number": None},
    "2_hissa": {"hissa_number": None},
    "3_khata_vara": {
        "total_extent": None,
        "phut_kharab_a": None,
        "phut_kharab_b": None,
        "remaining_extent": None,
    },
    "4_land_revenue": {
        "land_revenue": None,
        "jodi": None,
        "cess": None,
        "water_rate": None,
        "total_revenue": None,
    },
    "5_soil_type": {"soil_type": None},
    "6_land_classification": {"land_classification": None},
    "7_tree_details": {"tree_count": None, "tree_types": []},
    "8_irrigation_details": {
        "source": None,
        "kharif_extent": None,
        "rabi_extent": None,
        "garden_extent": None,
        "total_extent": None,
    },
    "9_ownership_details": [],
    "10_possession_details": {
        "possession_type": None,
        "owner_wise_possession": [],
    },
    "11_rights_liabilities": {
        "mr_number": None,
        "court_stay": False,
        "court_reference": None,
        "recorded_on": None,
        "rights": None,
        "liabilities": None,
        "remarks": None,
    },
    "12_cultivation_details": [],
    "13_other_rights": {
        "other_rights": None,
        "restrictions": None,
        "remarks": None,
    },
}


# Pre-printed form wording. A model that echoes one of these as a VALUE has
# copied the heading strip rather than reading an entry, which is exactly the
# failure the Tesseract mapper had. Compared after folding (see _fold_text).
#
# The guard MUST be field-scoped, not global. The same Kannada word is a
# heading in one block and a legitimate value in another — ಮುಂಗಾರು / ಹಿಂಗಾರು
# head the sub-columns of block 8, but they are exactly what block 12's
# `season` field is supposed to contain. A global set blanked every season on a
# correct read, so words like these live in _LABELS_BY_BLOCK (heading only
# where they really are a heading) rather than in _UNIVERSAL_LABELS.

# Never a valid value anywhere on the form: section headings, column headings
# and the browser chrome around the page.
_UNIVERSAL_LABELS = {
    "ಸರ್ವೆನಂಬರ", "ಹಿಸ್ಸಾ", "ಖಾತೆವಾರು", "ಕಂದಾಯ", "ಮಣ್ಣಿನನಮೂನೆ", "ಪಟ್ಟಾ",
    "ಮರಗಳಸಂಖ್ಯೆ", "ಹೆಸರು", "ಸಂಖ್ಯೆ", "ವಿಸ್ತೀರ್ಣ", "ಖಾತೆನಂ", "ಒಟ್ಟು",
    "ನೀರಾವರಿಮೂಲ", "ವರ್ಷ", "ಕಾಲ", "ಹಕ್ಕುಗಳು", "ಋಣಗಳು", "ಬೆಳೆಯಹೆಸರು",
    "ಭೂಕಂದಾಯ", "ಜೋಡಿ", "ಸೆಸ್ಸುಗಳು", "ನೀರಿನದರ", "ಉಳಿದದ್ದು", "ಪೂಟ್ಖರಾಬ್",
    "ಸಾಗುವಳಿದಾರರಹೆಸರು", "ಸ್ವಾಧೀನದಾರರಹೆಸರುಮತ್ತುವಿಳಾಸ",
    "ಕಬ್ಜೆಅಥವಾಸ್ವಾಧೀನತೆಯರೀತಿ", "ಭೂಮಿಯಉಪಯೋಗ", "ಸಾಗುವಳಿಪದ್ಧತಿ",
    "ಬೇಸಾಯಪ್ರಕಾರನೀರಾವರಿಯವಿಸ್ತೀರ್ಣ", "ಇತರೆಹಕ್ಕುಗಳುಮತ್ತುಋಣಗಳು",
    "first", "previous", "next", "last", "forviewingonly",
}

# Words that are a heading ONLY inside the named block. Block 8's sub-columns
# are headed by the season names; block 12 stores those same names as data.
_LABELS_BY_BLOCK = {
    "8_irrigation_details": {"ಮುಂಗಾರು", "ಹಿಂಗಾರು", "ಬಾಗಾಯ್ತು", "ಬೇಸಿಗೆ"},
}


def _is_quota_error(error) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _is_missing_model_error(error) -> bool:
    """A 404 NOT_FOUND (model unavailable to this key / misspelled) is
    permanent, not transient — retrying just burns the backoff budget."""
    message = str(error)
    return "404" in message or "NOT_FOUND" in message


def _is_spend_cap_error(error) -> bool:
    """A PROJECT-level 429 ("exceeded its monthly spending cap"), as opposed to
    a per-model daily quota.

    The distinction decides whether falling through to the next model is worth
    anything. A per-model quota is per-name, so the next model has its own
    bucket and the chain is useful. A spend cap belongs to the PROJECT and is
    shared by every model on the key — continuing the chain cannot succeed, it
    just spends 3 retries per remaining model before failing anyway. Callers
    abort the chain on this.
    """
    message = str(error).lower()
    return "spending cap" in message or "spend cap" in message


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


def _clean_str(value):
    """Normalise a scalar to a trimmed string, or None. Gemini sometimes
    returns the literal strings "null"/"N/A"/"-" for a blank box."""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    if not text or text.lower() in {"null", "none", "n/a", "na", "-", "--"}:
        return None

    return text


def _fold_text(value) -> str:
    """Fold a value for comparison only: strip whitespace, the ZWJ/ZWNJ joiners
    and the Kannada nukta that OCR and models sprinkle inconsistently, then
    lowercase. Two renderings of the same name fold equal, while a real
    character difference still does not."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("‌", "").replace("‍", "")
    text = re.sub(r"[\s.,|/\\_-]+", "", text)

    return text.lower()


def _is_form_label(value, block_key: str = None) -> bool:
    """True when `value` is just a pre-printed heading rather than an entry.

    `block_key` scopes the check: a word listed in _LABELS_BY_BLOCK counts as a
    heading only inside that block. Without it, only the universal headings
    match — so callers that cannot name a block stay conservative.
    """
    folded = _fold_text(value)

    if not folded:
        return False

    if folded in _UNIVERSAL_LABELS:
        return True

    return folded in _LABELS_BY_BLOCK.get(block_key, ())


def _strip_form_labels(parsed: dict) -> list:
    """Blank any scalar field whose value is just a pre-printed form label, and
    report which fields were blanked. This is the anti-label-copy guard; it is
    the one place we overrule the model, and only ever towards null."""
    stripped = []

    for block_key, block in parsed.items():
        if isinstance(block, dict):
            for field, value in list(block.items()):
                if _is_form_label(value, block_key):
                    block[field] = None
                    stripped.append(f"{block_key}.{field}")

        elif isinstance(block, list):
            for index, item in enumerate(block):
                if not isinstance(item, dict):
                    continue
                for field, value in list(item.items()):
                    if _is_form_label(value, block_key):
                        item[field] = None
                        stripped.append(f"{block_key}[{index}].{field}")

    return stripped


def _has_latin_in_kannada_field(value) -> bool:
    """True when a name-bearing field mixes Latin letters into Kannada text.
    That is the "ಕೋಂ -> BOLO" corruption signature: a genuine RTC name field is
    either all-Kannada or (rarely) an all-Latin transliteration, never both."""
    if not isinstance(value, str):
        return False

    has_kannada = bool(re.search(r"[ಀ-೿]", value))
    has_latin = bool(re.search(r"[A-Za-z]{2,}", value))

    return has_kannada and has_latin


_NAME_FIELDS = ("owner_name", "father_or_spouse_name", "cultivator_name")


def _normalize(parsed: dict) -> dict:
    """Coerce a raw model dict into the exact pipeline block contract: every
    block present, scalars cleaned, name lists well formed."""
    result = {}

    for key, empty in _EMPTY_BLOCKS.items():
        value = parsed.get(key)

        if isinstance(empty, list):
            result[key] = value if isinstance(value, list) else []
            continue

        block = dict(empty)

        if isinstance(value, dict):
            for field in empty:
                if field in value:
                    block[field] = value[field]

        result[key] = block

    # Scalars: trim, and map the model's stand-ins for "blank" onto real nulls.
    for key, block in result.items():
        if isinstance(block, dict):
            for field, value in block.items():
                if isinstance(value, (list, dict)):
                    continue
                block[field] = _clean_str(value)

        elif isinstance(block, list):
            for item in block:
                if not isinstance(item, dict):
                    continue
                for field, value in item.items():
                    if isinstance(value, (list, dict)):
                        continue
                    item[field] = _clean_str(value)

    # court_stay is a boolean flag, not a string — _clean_str would have turned
    # a literal False into None.
    rights = result["11_rights_liabilities"]
    rights["court_stay"] = bool(parsed.get("11_rights_liabilities", {}).get("court_stay")) \
        if isinstance(parsed.get("11_rights_liabilities"), dict) else False

    # 7_tree_details.tree_types must stay a list.
    trees = result["7_tree_details"]
    if not isinstance(trees.get("tree_types"), list):
        trees["tree_types"] = []

    # 10_possession_details.possession_type is a list of distinct kinds.
    possession = result["10_possession_details"]
    kinds = possession.get("possession_type")

    if isinstance(kinds, str):
        kinds = [kinds]
    if not isinstance(kinds, list):
        kinds = []

    # A model that sees two kinds stacked in the column often returns them as
    # one comma-joined string ("ಪೌತಿ, ಮೇಲಿನ ಜಂಟಿ"). Split so the field is a real
    # list of kinds, which is what callers filter on.
    split_kinds = []

    for kind in kinds:
        if isinstance(kind, str):
            split_kinds.extend(re.split(r"\s*[,;/]\s*", kind))
        else:
            split_kinds.append(kind)

    split_kinds = [
        k for k in (_clean_str(k) for k in split_kinds)
        if k and not _is_form_label(k, "10_possession_details")
    ]
    possession["possession_type"] = list(dict.fromkeys(split_kinds)) or None

    owner_wise = possession.get("owner_wise_possession")
    possession["owner_wise_possession"] = owner_wise if isinstance(owner_wise, list) else []

    # Drop ownership / cultivation items that carry no identifying content at
    # all — a model asked for an array sometimes returns one all-null template
    # item for a blank section.
    result["9_ownership_details"] = [
        owner for owner in result["9_ownership_details"]
        if isinstance(owner, dict) and any(owner.get(f) for f in owner)
    ]

    result["12_cultivation_details"] = [
        row for row in result["12_cultivation_details"]
        if isinstance(row, dict) and any(row.get(f) for f in row)
    ]

    return result


def _collect_names(parsed: dict) -> list:
    """Every name-bearing value in the read, for the Latin-corruption check."""
    names = []

    for owner in parsed.get("9_ownership_details", []):
        if isinstance(owner, dict):
            names.extend(owner.get(f) for f in _NAME_FIELDS if owner.get(f))

    for row in parsed.get("12_cultivation_details", []):
        if isinstance(row, dict):
            names.extend(row.get(f) for f in _NAME_FIELDS if row.get(f))

    for pair in parsed.get("10_possession_details", {}).get("owner_wise_possession", []):
        if isinstance(pair, dict) and pair.get("owner_name"):
            names.append(pair["owner_name"])

    return [n for n in names if isinstance(n, str)]


# Header + single-value blocks worth cross-checking between two models. These
# are the fields the property-record save and the completeness score read, so a
# silent disagreement here is the most expensive kind.
_VERIFY_SCALARS = [
    ("header", "taluk"),
    ("header", "hobli"),
    ("header", "village"),
    ("header", "valid_from"),
    ("1_survey_number", "survey_number"),
    ("2_hissa", "hissa_number"),
    ("3_khata_vara", "total_extent"),
    ("3_khata_vara", "remaining_extent"),
    ("4_land_revenue", "total_revenue"),
    ("11_rights_liabilities", "mr_number"),
    ("11_rights_liabilities", "recorded_on"),
]


def _diff_reads(primary: dict, secondary: dict) -> dict:
    """Compare two independent reads. This CANNOT prove a value right — two
    models can repeat the same misread — but a disagreement is a reliable
    signal that a human should look. Mirrors the MR extractor's approach,
    which RTC needs because the form has no printed total to checksum."""
    scalar_disagreements = []

    for block_key, field in _VERIFY_SCALARS:
        a = primary.get(block_key, {}).get(field)
        b = secondary.get(block_key, {}).get(field)

        if _fold_text(a) != _fold_text(b):
            scalar_disagreements.append({
                "field": f"{block_key}.{field}",
                "primary": a,
                "secondary": b,
            })

    owner_disagreements = []
    primary_owners = primary.get("9_ownership_details", [])
    secondary_owners = secondary.get("9_ownership_details", [])

    if len(primary_owners) != len(secondary_owners):
        owner_disagreements.append({
            "field": "9_ownership_details.count",
            "primary": len(primary_owners),
            "secondary": len(secondary_owners),
        })
    else:
        for index, (a, b) in enumerate(zip(primary_owners, secondary_owners)):
            for field in ("owner_name", "father_or_spouse_name", "extent", "khata_no"):
                if _fold_text(a.get(field)) != _fold_text(b.get(field)):
                    owner_disagreements.append({
                        "field": f"9_ownership_details[{index}].{field}",
                        "primary": a.get(field),
                        "secondary": b.get(field),
                    })

    cultivation_disagreements = []
    primary_rows = primary.get("12_cultivation_details", [])
    secondary_rows = secondary.get("12_cultivation_details", [])

    if len(primary_rows) != len(secondary_rows):
        cultivation_disagreements.append({
            "field": "12_cultivation_details.count",
            "primary": len(primary_rows),
            "secondary": len(secondary_rows),
        })

    return {
        "scalar_disagreements": scalar_disagreements,
        "owner_disagreements": owner_disagreements,
        "cultivation_disagreements": cultivation_disagreements,
    }


def _score(parsed: dict) -> int:
    """Rank a read by how much of the form it actually recovered, so that when
    two models both answer we keep the fuller one. Corrupted names are
    penalised heavily — a read that reintroduces the Latin-in-Kannada bug is
    worse than a read that returned null there."""
    score = 0

    for block_key, field in _VERIFY_SCALARS:
        if parsed.get(block_key, {}).get(field):
            score += 3

    score += 4 * len(parsed.get("9_ownership_details", []))
    score += 2 * len(parsed.get("12_cultivation_details", []))

    for name in _collect_names(parsed):
        if _has_latin_in_kannada_field(name):
            score -= 10

    return score


def _try_model(client, contents, model):
    """
    Call a SINGLE model, retrying transient failures with exponential backoff.
    A daily-quota 429 is NOT transient, so stop early. Returns
    (text_or_None, last_error).
    """
    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=GENERATION_CONFIG,
            )
        except Exception as e:
            last_error = e
            print(f"Gemini RTC {model} attempt {attempt + 1} failed:", e)
            if _is_quota_error(e) or _is_missing_model_error(e):
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

        if response and getattr(response, "text", None):
            return response.text, last_error

    return None, last_error


def _read_once(client, contents, model):
    """Run ONE model and normalise its output. Returns (read_or_None, error)."""
    text, error = _try_model(client, contents, model)

    if not text:
        return None, error

    parsed = _extract_json(text)

    if not isinstance(parsed, dict) or not parsed:
        return None, error

    normalized = _normalize(parsed)
    stripped = _strip_form_labels(normalized)

    return {
        "model": model,
        "data": normalized,
        "stripped_form_labels": stripped,
        "score": _score(normalized),
    }, error


def _unavailable_result(error) -> dict:
    """Shape returned when NO model produced a read (e.g. exhausted quota). The
    caller falls back to the cell-OCR mapper; this keeps the block contract so
    nothing downstream has to special-case it."""
    result = {"document_type": "RTC"}

    for key, empty in _EMPTY_BLOCKS.items():
        result[key] = list(empty) if isinstance(empty, list) else dict(empty)

    result.update({
        "extraction_status": "requires_manual_review",
        "text_verification": {"status": "unavailable", "verified_by": None},
        "disagreements": {},
        "warnings": [
            "Gemini API quota exhausted — no model could read the document."
            if _is_quota_error(error) else
            "No Gemini model returned a usable read for this document."
        ],
        "error": str(error) if error is not None else None,
        "extraction_source": "gemini_vision_rtc_v1:unavailable",
    })

    return result


def extract_rtc_gemini(file_path: str, ocr_text: str = None):
    """
    Extract a Karnataka RTC / Pahani (Form No. 2) from the document image with
    Gemini vision, grounded by the pipeline's OCR text.

    Returns the 13-block structure the RTC pipeline contract expects, plus
    verification metadata (text_verification, disagreements, warnings) so the
    caller can tell a confident read from one needing review.
    """
    if not settings.GOOGLE_API_KEY:
        return _unavailable_result(
            RuntimeError("GOOGLE_API_KEY is missing in .env")
        )

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    image = _load_image(file_path)

    contents = [RTC_PROMPT, image]

    # Ground the read with the pipeline OCR text when we have it. It is noisy
    # on Kannada (that is why this module exists), so it is offered strictly as
    # a hint for Latin/numeric tokens — the image stays authoritative.
    if ocr_text:
        contents.append(
            "Reference OCR text from the same page. It is UNRELIABLE for"
            " Kannada script and may contain Latin letters where Kannada was"
            " printed. Use it only to confirm digits, dates and Latin tokens"
            " such as the MR number. Where it conflicts with the image, TRUST"
            " THE IMAGE.\n\n" + ocr_text[:6000]
        )

    reads = []
    last_error = None

    # First pass: walk the model list until one answers. For a per-model daily
    # quota a 429 on the leader costs nothing but a skip. A PROJECT spend cap is
    # different — every model shares it, so the chain is abandoned immediately
    # rather than spending 3 retries per remaining name on a certain failure.
    for model in RTC_MODELS:
        read, error = _read_once(client, contents, model)

        if error is not None:
            last_error = error

        if read:
            reads.append(read)
            break

        if error is not None and _is_spend_cap_error(error):
            break

    if not reads:
        return _unavailable_result(last_error)

    # Second pass: an independent read from a DIFFERENT model. An RTC has no
    # printed arithmetic to checksum, so cross-model agreement is the only
    # available trust signal. A failure here is not fatal — we degrade to an
    # unverified single read rather than losing the extraction.
    for model in RTC_MODELS:
        if model == reads[0]["model"]:
            continue

        read, error = _read_once(client, contents, model)

        if error is not None:
            last_error = error

        if read:
            reads.append(read)
            break

        if error is not None and _is_spend_cap_error(error):
            break

    reads.sort(key=lambda r: r["score"], reverse=True)

    best = reads[0]
    data = best["data"]

    warnings = []

    if best["stripped_form_labels"]:
        warnings.append(
            "Blanked pre-printed form labels returned as values: "
            + ", ".join(best["stripped_form_labels"])
        )

    corrupted = [
        name for name in _collect_names(data)
        if _has_latin_in_kannada_field(name)
    ]

    if corrupted:
        warnings.append(
            "Name fields mix Latin letters into Kannada text and may be"
            " misread: " + "; ".join(corrupted)
        )

    if len(reads) > 1:
        other = reads[1]
        disagreements = _diff_reads(data, other["data"])

        total = sum(len(v) for v in disagreements.values())

        text_verification = {
            "status": "verified" if total == 0 else "disagreement",
            "verified_by": other["model"],
        }

        if total:
            warnings.append(
                f"{total} field(s) differ between {best['model']} and"
                f" {other['model']} — verify against the document."
            )
    else:
        disagreements = {}
        text_verification = {"status": "unavailable", "verified_by": None}
        warnings.append(
            "Only one model returned a read, so no cross-model verification"
            " was possible."
        )

    result = {"document_type": "RTC"}
    result.update(data)

    result.update({
        "extraction_status": (
            "extracted" if text_verification["status"] == "verified"
            else "extracted_needs_review"
        ),
        "text_verification": text_verification,
        "disagreements": disagreements,
        "warnings": warnings,
        "error": None,
        "extraction_source": f"gemini_vision_rtc_v1:{best['model']}",
    })

    return result

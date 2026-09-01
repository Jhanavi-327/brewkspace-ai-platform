"""
OCR-text fallback for Patta Book / Khata Extract documents.

Used ONLY when Gemini vision is unavailable (e.g. an exhausted API quota). It
mirrors the Mutation Register fallback (`mr_ocr_fallback.py`): recover the values
this pipeline's OCR prints RELIABLY, and refuse to invent the ones it mangles.

A Patta's holder table OCRs far better than an MR's (it is a clean
landrecords.karnataka.gov.in print), but the Kannada HOLDER NAMES still fuse and
split across lines ("ತಿಮ" / "ಅ" / "ಕ್ಕ" for ತಿಮ್ಮಕ್ಕ; a holder's second name
wraps onto its own line). Putting a silently-wrong owner into a title record is
worse than omitting it, so this module does NOT reconstruct per-holder rows —
"entries" is always EMPTY and flagged for manual review.

What it DOES recover, because the OCR prints it reliably:
  * the labelled header — khata number, district, taluk, hobli, village
  * the footer — place (ಮೊಕ್ಕಾಂ), date (ದಿನಾಂಕ), amount (ಮೊತ್ತ)
  * the ಒಟ್ಟು grand-total row's numbers (extent + the money columns)
  * the survey/hissa codes, which have a distinctive ASCII shape

Every value here is lower-confidence than a Gemini read; the caller marks it so.
"""

import re


# Tesseract renders the anusvara ಂ (U+0C82) as the Kannada digit ೦ (U+0CE6). A
# digit can never appear inside a Kannada word, so a ೦ directly after a Kannada
# letter is always a misread anusvara.
_ANUSVARA_FIX_RE = re.compile(r"(?<=[ಀ-೥])೦")

_SEPARATORS = " \t:-~.`>_,;|ಃ"


def _clean(value):
    if value is None:
        return None
    value = str(value).translate({0x200B: None, 0x200C: None, 0x200D: None, 0x00A0: None})
    value = _ANUSVARA_FIX_RE.sub("ಂ", value)
    value = re.sub(r"\s+", " ", value).strip(_SEPARATORS + " ")
    return value or None


# Header field -> label variants the OCR prints for it. EasyOCR routinely mangles
# the label (ಖಾತೆ -> ಕ್ರೆ / ಜಾತ, ಗ್ರಾಮ -> ಗರಾಮ) and wraps the VALUE onto the next
# line, so each field is matched by a stem the engines agree on.
_HEADER_LABELS = {
    "district": ("ಜಿಲ್ಲೆ", "ಜಿಲ್ಲ"),
    "taluk": ("ತಾಲ್ಲೂಕು", "ತಾಲ್ಕೂಕು", "ತಾಲ್ಲೂಕ"),
    "hobli": ("ಹೋಬಳಿ", "ಹೋಬ"),
    "village": ("ಗ್ರಾಮ", "ಗರಾಮ"),
}

# Any header label — so a value-search never swallows the NEXT field's label.
_ANY_HEADER_LABEL = re.compile(
    "|".join(("ಜಿಲ್ಲೆ", "ಜಿಲ್ಲ", "ತಾಲ್ಲೂಕು", "ತಾಲ್ಕೂಕು", "ಹೋಬಳಿ", "ಹೋಬ",
              "ಗ್ರಾಮ", "ಗರಾಮ", "ಸರ್ವೆ", "ಕ್ರಮ", "ಸಂಖ್ಯೆ", "ವಿಸ್ತೀ", "ಆಕಾರ",
              "ನಂಬರ್", "ಸಂಬರ್"))
)

_KANNADA_RE = re.compile(r"[ಀ-೿]")

# ಖಾತೆ ನಂಬರ್ : N — the khata number. OCR mangles ಖಾತೆ (ಕ್ರೆ / ಜಾತ / ಖಾತ) but keeps
# "ನಂಬರ್" and the digits after the colon; the survey column's ಸರ್ವೆ ... ನಂ has no
# ": <digits>" on the same token, so anchoring on ನಂಬರ್ + colon + digits is safe.
_KHATA_RE = re.compile(r"ನಂಬರ್\W{0,4}([0-9]{1,5})\b")

# Footer labels — value printed on the SAME line after the colon.
_PLACE_RE = re.compile(r"ಮೊಕ್ಕಾ[ಂ೦]\W*([^\n]+)")
_DATE_RE = re.compile(r"ದಿನಾ[ಂ೦]ಕ\W*([0-9]{1,4}[/\-][0-9]{1,2}[/\-][0-9]{2,4})")
_AMOUNT_RE = re.compile(r"ಮೊತ್ತ\W*([0-9][0-9,]*)")

# The ಒಟ್ಟು grand-total row (OCR: ಒಟ್ಟು / ಒಪ್ಮು / ಒಲ್ಮು), followed by an extent and
# the money figures.
_TOTAL_LABEL_RE = re.compile(r"ಒ[ಟಪಲ]್[ಟಮ]ು")
_EXTENT_RE = re.compile(r"\b\d{1,3}\.\d{2}\.\d{2}\.\d{2}\b")
_DECIMAL_RE = re.compile(r"\b\d{1,4}\.\d{1,2}\b")

# A Patta survey/hissa code: digits, a "*" hissa, "/" separators. OCR reads "/"
# as "|", "I" or a space and sometimes drops one, so match loosely then normalise.
_SURVEY_RE = re.compile(r"\b\d{1,4}\s*[/|I]\s*\*[\s/|I]*[\d*]{0,4}")

# Footer/print-chrome lines that must never be taken as a header value.
_CHROME_RE = re.compile(r"landrecords|karnataka\.gov|https?|Print|MR_Preview|\d{1,2}[:.]\d{2}\s*[AP]M", re.I)


def _engine_lines(ocr_text: str) -> list:
    """Non-blank lines, with print-chrome and pure-punctuation noise dropped."""
    lines = []
    for raw in ocr_text.splitlines():
        line = raw.strip()
        if not line or _CHROME_RE.search(line):
            continue
        lines.append(line)
    return lines


def _is_value_line(text: str) -> bool:
    """A real header value carries Kannada letters and is not itself a label."""
    if not text or _ANY_HEADER_LABEL.search(text):
        return False
    letters = _KANNADA_RE.findall(text)
    return len(letters) >= 2


def _header_value(lines: list, variants) -> str:
    """
    Value for a header label. The value sits AFTER the label on the same line
    (Tesseract) or on one of the next lines (EasyOCR wraps it), so try both.
    """
    for index, line in enumerate(lines):
        for variant in variants:
            position = line.find(variant)
            if position == -1:
                continue

            tail = _clean(line[position + len(variant):])
            if tail and _is_value_line(tail):
                return tail

            # Value wrapped onto a following line.
            for next_line in lines[index + 1: index + 4]:
                candidate = _clean(next_line)
                if candidate and _is_value_line(candidate):
                    return candidate
            break
    return None


def _khata_number(ocr_text: str) -> str:
    """First ನಂಬರ್ : <digits> in the header block."""
    match = _KHATA_RE.search(ocr_text)
    return match.group(1) if match else None


def _total_row(lines: list) -> dict:
    """
    The ಒಟ್ಟು grand-total figures. From the total-label line onward, take the
    dotted extent and the following decimals (akara, local, health, edu, total).
    Numbers are the OCR's most stable output, and the caller can reconcile them.
    """
    for index, line in enumerate(lines):
        if not _TOTAL_LABEL_RE.search(line):
            continue

        window = " ".join(lines[index: index + 8])
        extent_match = _EXTENT_RE.search(window)
        # Decimals AFTER the extent are the money columns.
        after = window[extent_match.end():] if extent_match else window
        decimals = _DECIMAL_RE.findall(after)

        row = {}
        if extent_match:
            row["extent"] = extent_match.group(0)
        fields = ("akara", "local_cess", "health_cess", "education_cess", "total")
        for field, value in zip(fields, decimals):
            row[field] = value
        if row:
            return row
    return {}


def _survey_numbers(lines: list) -> list:
    """Distinct survey/hissa codes, separators normalised back to '/'."""
    surveys = []
    for line in lines:
        for match in _SURVEY_RE.findall(line):
            code = re.sub(r"[|I]", "/", match)
            code = re.sub(r"\s+", "", code)
            code = re.sub(r"/{2,}", "/", code)
            if code and code not in surveys:
                surveys.append(code)
    return surveys


def extract_patta_book_from_ocr(ocr_text: str) -> dict:
    """
    Best-effort recovery of a Patta Book's labelled fields from the OCR text.
    Returns a dict of what it could recover (missing fields simply absent), plus
    "total_row" and "affected_survey_numbers". NEVER returns holder entries.
    """
    if not ocr_text or not ocr_text.strip():
        return {}

    lines = _engine_lines(ocr_text)
    recovered = {}

    khata = _khata_number(ocr_text)
    if khata:
        recovered["khata_number"] = khata

    for field, variants in _HEADER_LABELS.items():
        value = _header_value(lines, variants)
        if value:
            recovered[field] = value

    place = _PLACE_RE.search(ocr_text)
    if place:
        cleaned = _clean(place.group(1))
        if cleaned:
            recovered["place"] = cleaned

    date = _DATE_RE.search(ocr_text)
    if date:
        recovered["date"] = date.group(1)

    amount = _AMOUNT_RE.search(ocr_text)
    if amount:
        recovered["amount"] = amount.group(1)

    total_row = _total_row(lines)
    if total_row:
        recovered["total_row"] = total_row

    surveys = _survey_numbers(lines)
    if surveys:
        recovered["affected_survey_numbers"] = surveys

    return recovered

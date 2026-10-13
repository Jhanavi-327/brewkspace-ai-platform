"""
Per-cell OCR of a Mutation Register's party table.

Used ONLY when Gemini vision is unavailable. Reuses the generic OpenCV grid
detection already in the RTC services (line detection + cell boxes + row
grouping) — those functions are layout-agnostic; nothing in RTC is modified.

Cropping each grid cell and OCR-ing it separately recovers the table STRUCTURE
reliably (columns stay separate, names cannot fuse with extents). The VALUES are
another matter: on low-resolution scans Tesseract misreads digits badly
(60.21.00.00 -> 0.21.00.00, 0.35.00.00 -> 35.00.00) and destroys survey codes
(59/*/* -> "Sl i").

So this module NEVER corrects or reformats a cell. It reports exactly what OCR
saw, keeps the raw cell text alongside, and the caller marks every field as
low-confidence needing review. A "corrected" extent would be a fabrication.
"""

import re
from collections import Counter

import fitz
import numpy as np
import pytesseract
from PIL import Image

from app.services.rtc_table_service import (
    preprocess_image,
    detect_table_grid,
    extract_cell_boxes,
    group_cells_into_rows,
)
from app.services.rtc_cell_ocr_service import ocr_cell, preprocess_crop


# ಸರ್ವೆ ನಂಬರು | ಸ್ವಾಧೀನದಾರರ ಹೆಸರು | ವಿಸ್ತೀರ್ಣ | ಖಾತೆ ನಂಬರು |
# ಸ್ವಾಧೀನತೆಯ ರೀತಿ | ಹಕ್ಕುಗಳು | ಋಣಗಳು | ಸ್ಥಿತಿ
_POSSESSOR_COLUMNS = (
    "survey_hissa_number",
    "rights_changed_to",
    "extent_to",
    "khata_number",
    "possession_mode",
    "rights",
    "liabilities",
    "row_status",
)

# ಬಾಧಿತವಾದ ಸರ್ವೆ ಮತ್ತು ಹಿಸ್ಸಾ ನಂ. | ಮಾಡಿದವರು | ವಿಸ್ತೀರ್ಣ |
# ಪಡೆದವರು | ವಿಸ್ತೀರ್ಣ | ತಪಾಸಣೆ ಅಧಿಕಾರಿ
_COMMON_COLUMNS = (
    "survey_hissa_number",
    "rights_changed_from",
    "extent_from",
    "rights_changed_to",
    "extent_to",
    "inspecting_officer",
)

# Header fragments that survive OCR damage well enough to identify the layout.
_POSSESSOR_HINTS = ("ಧೀನದಾರರ", "ಖಾತೆ", "ಸ್ಥಿತಿ")
_COMMON_HINTS = ("ಮಾಡಿದವರು", "ಪಡೆದವರು", "ಬಾಧಿತ")

_MIN_TABLE_COLUMNS = 5


def _load_page_image(file_path: str):
    """First page of a PDF, or the image itself. Returns an RGB numpy array."""
    if file_path.lower().endswith(".pdf"):
        pdf = fitz.open(file_path)
        pix = pdf[0].get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf.close()
        return np.array(image)
    return np.array(Image.open(file_path).convert("RGB"))


def _pick_table_rows(rows):
    """
    The party table is the widest RECURRING row shape on the page — a header row
    plus at least one data row sharing the same column count. Returns
    (header_row, data_rows) or (None, []).
    """
    widths = Counter(len(row) for row in rows if len(row) >= _MIN_TABLE_COLUMNS)
    if not widths:
        return None, []

    # Prefer the widest shape that repeats; ties break toward more columns.
    width = max(widths, key=lambda w: (widths[w], w))
    group = [row for row in rows if len(row) == width]

    if len(group) < 2:
        return None, []
    return group[0], group[1:]


def _detect_layout(header_texts, column_count: int):
    joined = " ".join(header_texts)

    if any(hint in joined for hint in _POSSESSOR_HINTS):
        return "possessor"
    if any(hint in joined for hint in _COMMON_HINTS):
        return "common"

    # OCR destroyed the header labels — fall back to the column count.
    if column_count >= 8:
        return "possessor"
    if column_count == 6:
        return "common"
    return None


def _blank_to_none(text):
    text = (text or "").strip()
    return text or None


# Columns holding only numbers. Reading them with a digit whitelist stops
# Tesseract turning "0" into ")" or hallucinating letters.
_NUMERIC_COLUMNS = {
    "extent_from", "extent_to", "khata_number", "revenue", "patta_number",
}
_NUMERIC_WHITELIST = "0123456789."
_SURVEY_WHITELIST = "0123456789/*P"


def _crop(image, cell):
    """
    Cell pixels with NO inset.

    The RTC helper insets 2px, which on this table clips the left stroke of the
    first glyph and the leading digit of an extent — turning ಸರ್ಕಾರಿ into
    ನರ್ತಾರಿ and 60.21.00.00 into 0.21.00.00. Use the exact cell bounds instead.
    """
    y1 = max(0, cell["y"])
    y2 = min(image.shape[0], cell["y2"])
    x1 = max(0, cell["x"])
    x2 = min(image.shape[1], cell["x2"])
    return image[y1:y2, x1:x2]


def _run_ocr(crop, lang: str, psm: str, whitelist: str = None) -> str:
    if crop.size == 0:
        return ""
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    try:
        text = pytesseract.image_to_string(
            preprocess_crop(crop), lang=lang, config=config
        )
    except Exception:
        return ""
    return " ".join(text.split())


def _ocr_field_cell(image, cell, field: str) -> str:
    """OCR one cell, tuned to what the column actually contains."""
    crop = _crop(image, cell)

    if field in _NUMERIC_COLUMNS:
        return _run_ocr(crop, "eng", "7", _NUMERIC_WHITELIST)

    if field == "survey_hissa_number":
        # Digits/asterisks read cleanly; the thin slashes often do not.
        return _run_ocr(crop, "eng", "7", _SURVEY_WHITELIST)

    return _run_ocr(crop, "kan+eng", "6")


def _drop_container_cells(header_row):
    """
    Grid detection sometimes returns an outer box spanning the whole table
    alongside the real header cells. It would swallow every column band, so drop
    any header cell wider than half the header's total span.
    """
    if not header_row:
        return header_row

    left = min(cell["x"] for cell in header_row)
    right = max(cell["x2"] for cell in header_row)
    span = max(1, right - left)

    kept = [c for c in header_row if (c["x2"] - c["x"]) <= span * 0.5]
    return kept or header_row


# Per-column band settings, tuned against real documents:
#   left_pad  – how far to expand LEFT of the detected border. The border sits
#               ON the first glyph, so text columns need a few px back or the
#               leading letter/digit is destroyed (ಬಿ->ಶಿ, 0.07->2.07). The FIRST
#               column must not expand: there is only the table frame to its left.
#   psm 4     – one text line per row, which keeps the dots in "0.07.00.00";
#               psm 6 merges them into "0070000".
_BAND_SETTINGS = {
    "survey": dict(left_pad=0, lang="eng", psm="6", whitelist="0123456789/*B"),
    "extent": dict(left_pad=2, lang="eng", psm="4", whitelist=_NUMERIC_WHITELIST),
    "text": dict(left_pad=6, lang="kan+eng", psm="6", whitelist=None),
}

# Which band setting each common-layout column uses.
_COLUMN_KINDS = {
    "survey_hissa_number": "survey",
    "extent_from": "extent",
    "extent_to": "extent",
}


def _column_lines(image, cell, top: int, bottom: int, kind: str = "text"):
    """
    OCR a whole column band beneath the header as MULTI-LINE text.

    The common layout draws no horizontal rules between party rows — several
    names are stacked inside one tall cell — so per-cell OCR cannot split them.
    Reading the band and splitting on lines recovers each printed line.
    """
    settings = _BAND_SETTINGS.get(kind, _BAND_SETTINGS["text"])

    x1 = max(0, cell["x"] - settings["left_pad"])
    x2 = min(image.shape[1], cell["x2"] + 2)
    crop = image[top:bottom, x1:x2]
    if crop.size == 0:
        return []

    config = f"--oem 3 --psm {settings['psm']}"
    if settings["whitelist"]:
        config += f" -c tessedit_char_whitelist={settings['whitelist']}"

    try:
        text = pytesseract.image_to_string(
            preprocess_crop(crop), lang=settings["lang"], config=config
        )
    except Exception:
        return []

    return [line.strip() for line in text.splitlines() if line.strip()]


# An extent prints as dotted groups; a survey code carries "/" with * or a hissa.
_EXTENT_RE = re.compile(r"\d+[.,]\d+[.,]\d+")
_SURVEY_RE = re.compile(r"\d{1,4}\s*/\s*[*\w]")
# ಋಣಗಳು, an MR reference, or a transfer-restriction note. OCR mangles the
# restriction wording heavily (ನಿಷೇಧಿಸಿದೆ -> ವಿಶೇದಿಸಿದೆ, ಪರಭಾರೆ -> ಉರಭಾರ), so match
# on the stable fragments.
_LIABILITY_HINTS = (
    "ಣಗಳ", "MR ", "ನಿಷೇಧ", "ನಿಷೇದ", "ವಿಷೇದ", "ವಿಶೇದ",
    "ಪರಭಾರ", "ಪರಬಾರ", "ಉರಭಾರ", "ಯರಭಾರ", "ವರಭಾರ",
)
_RELATION_HINT = "ಸಂಬಂಧ"
_RELATION_HINT_ALT = "ಸ೦ಬ೦ಧ"


def _classify_line(line: str) -> str:
    if _RELATION_HINT in line or _RELATION_HINT_ALT in line:
        return "relation"
    if any(hint in line for hint in _LIABILITY_HINTS):
        return "liabilities"
    if _SURVEY_RE.search(line):
        return "survey"
    if _EXTENT_RE.search(line):
        return "extent"
    # A name needs some Kannada; otherwise it is OCR noise.
    if sum(1 for ch in line if "ಀ" <= ch <= "೿") >= 3:
        return "name"
    return "noise"


# A full survey/hissa code, used to trim OCR punctuation noise off the line.
_SURVEY_TOKEN_RE = re.compile(r"\d{1,4}\s*/\s*[*\w]+(?:\s*/\s*[*\w]+)?")


def _split_column(lines):
    """Group a column's OCR lines by what each line actually is."""
    grouped = {"survey": [], "extent": [], "name": [], "liabilities": [], "relation": []}
    for line in lines:
        kind = _classify_line(line)
        if kind == "noise":
            continue
        if kind == "survey":
            # Keep just the code; drop trailing OCR punctuation ("139/*/3 ;").
            token = _SURVEY_TOKEN_RE.search(line)
            line = token.group(0).replace(" ", "") if token else line
        grouped[kind].append(line)
    return grouped


def extract_mr_table_from_image(file_path: str) -> dict:
    """
    Returns {"layout", "entries", "raw_rows"}. `entries` values are RAW OCR text —
    uncorrected and unreliable, especially numbers. Empty dict-ish result when no
    table is found. Never raises for a missing/odd table.
    """
    image = _load_page_image(file_path)

    thresh = preprocess_image(image)
    grid = detect_table_grid(thresh)
    cells = extract_cell_boxes(grid, image.shape[1], image.shape[0])
    rows = group_cells_into_rows(cells)

    header_row, data_rows = _pick_table_rows(rows)

    if header_row and data_rows:
        # Ruled table (the possessor layout): every party row has its own cells.
        header_texts = [ocr_cell(image, cell) for cell in header_row]
        layout = _detect_layout(header_texts, len(header_row))

        if layout == "possessor":
            entries = []
            raw_rows = [header_texts]
            for row in data_rows:
                entry = {}
                texts = []
                for index, cell in enumerate(row):
                    if index >= len(_POSSESSOR_COLUMNS):
                        break
                    field = _POSSESSOR_COLUMNS[index]
                    text = _ocr_field_cell(image, cell, field)
                    texts.append(text)
                    entry[field] = _blank_to_none(text)
                raw_rows.append(texts)
                if any(entry.values()):
                    entries.append(entry)
            return {
                "layout": "possessor",
                "entries": entries,
                "columns": {},
                "raw_rows": raw_rows,
            }

    # Unruled table (the common layout): party rows share one tall cell per
    # column, so read each column band as multi-line text instead.
    return _extract_common_columns(image, rows)


def _extract_common_columns(image, rows) -> dict:
    header_row = next(
        (row for row in rows if len(row) >= _MIN_TABLE_COLUMNS), None
    )
    if not header_row:
        return {"layout": None, "entries": [], "columns": {}, "raw_rows": []}

    header_row = _drop_container_cells(sorted(header_row, key=lambda c: c["x"]))
    header_texts = [ocr_cell(image, cell) for cell in header_row]

    if _detect_layout(header_texts, len(header_row)) != "common":
        return {"layout": None, "entries": [], "columns": {}, "raw_rows": [header_texts]}

    top = max(cell["y2"] for cell in header_row)
    bottom = max(cell["y2"] for row in rows for cell in row)
    if bottom - top < 10:
        return {"layout": None, "entries": [], "columns": {}, "raw_rows": [header_texts]}

    columns = {}
    raw_rows = [header_texts]

    for index, cell in enumerate(header_row):
        if index >= len(_COMMON_COLUMNS):
            break
        field = _COMMON_COLUMNS[index]
        lines = _column_lines(image, cell, top, bottom, _COLUMN_KINDS.get(field, "text"))
        raw_rows.append(lines)
        columns[field] = _split_column(lines)

    survey_lines = columns.get("survey_hissa_number", {}).get("survey", [])
    from_names = columns.get("rights_changed_from", {}).get("name", [])
    to_names = columns.get("rights_changed_to", {}).get("name", [])
    from_extents = columns.get("extent_from", {}).get("extent", [])
    to_extents = columns.get("extent_to", {}).get("extent", [])
    liabilities = columns.get("rights_changed_to", {}).get("liabilities", [])

    # A single party's cell text wraps onto several lines, which would read as
    # several parties. Collapse them only when TWO independent signals agree that
    # the table has one party: exactly one extent AND at most one survey number.
    #
    # The extent count alone is not enough — on a low-resolution scan the extents
    # under-read, and trusting that count merged three real parties into one.
    extent_count = max(len(from_extents), len(to_extents))
    single_party = extent_count == 1 and len(survey_lines) <= 1

    if single_party:
        from_names = _rejoin_wrapped_names(from_names)
        to_names = _rejoin_wrapped_names(to_names)

    entries = _pair_common_entries(
        survey_lines, from_names, from_extents, to_names, to_extents
    )

    # With a single party, any ಋಣಗಳು / restriction lines in its block belong to it.
    if len(entries) == 1 and liabilities:
        entries[0]["liabilities"] = " ".join(liabilities)

    return {
        "layout": "common",
        "entries": entries,
        "columns": {
            "affected_survey_numbers": survey_lines,
            "rights_changed_from": from_names,
            "rights_changed_to": to_names,
            "extent_from": from_extents,
            "extent_to": to_extents,
            "liabilities": liabilities,
        },
        "raw_rows": raw_rows,
    }


def _rejoin_wrapped_names(names):
    """
    Join one party's wrapped name fragments ("ಚನ್ನಮ್ಮ ಕೋಂ" / "ಕೆ.ಮುನಿಶಾಮಪ್ಪ") back
    into a single name. Only called once the caller has established that the table
    really does hold a single party — with several parties the line-to-party
    boundaries are unknowable and the lines must be left alone.
    """
    return [" ".join(names)] if len(names) > 1 else names


def _at(values, index):
    """Value at `index` only when the list lines up one-per-entry."""
    return values[index] if index < len(values) else None


def _pair_common_entries(surveys, from_names, from_extents, to_names, to_extents):
    """
    One entry per printed party in the unruled common table.

    A column is only zipped positionally when its length MATCHES the party count
    — otherwise the lines don't correspond one-to-one and pairing them would
    silently attach the wrong extent to the wrong person. Mismatched columns are
    left null; the full lists still travel in "columns".
    """
    count = max(len(from_names), len(to_names))
    if not count:
        return []

    use_survey = len(surveys) == count
    use_from_extent = len(from_extents) == count
    use_to_extent = len(to_extents) == count

    entries = []
    for index in range(count):
        entries.append(
            {
                "survey_hissa_number": _at(surveys, index) if use_survey else None,
                "rights_changed_from": _at(from_names, index),
                "extent_from": _at(from_extents, index) if use_from_extent else None,
                "rights_changed_to": _at(to_names, index),
                "extent_to": _at(to_extents, index) if use_to_extent else None,
            }
        )
    return entries

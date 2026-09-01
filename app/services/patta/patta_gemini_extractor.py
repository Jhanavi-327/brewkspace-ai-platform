import itertools
import json
import re
import time

import fitz
from PIL import Image
from google import genai
from google.genai import types

from app.core.config import settings
from app.services.patta.patta_ocr_fallback import extract_patta_book_from_ocr


# Deterministic decoding: this is an extraction task, so we want the same
# document to yield the same output on every run (no sampling variance).
GENERATION_CONFIG = types.GenerateContentConfig(temperature=0)


PATTA_PROMPT = """
You are extracting fields from a Karnataka "Khata Extract / Patta Book copy"
(ಖಾತೆ ಪ್ರತಿ / ಪಟ್ಟಾ ಪುಸ್ತಕದ ಪ್ರತಿ) issued from landrecords.karnataka.gov.in.

Return ONLY valid JSON. No markdown. No explanation.

The document has a header block, one table of land holders, a total row,
and a footer. Extract exactly this structure:

{
  "khata_number": null,
  "district": null,
  "taluk": null,
  "hobli": null,
  "village": null,
  "entries": [
    {
      "serial_number": null,
      "survey_hissa_number": null,
      "holder_name": null,
      "father_name": null,
      "extent": null,
      "akara": null,
      "local_cess": null,
      "health_cess": null,
      "education_cess": null,
      "total": null
    }
  ],
  "total_row": {
    "extent": null,
    "akara": null,
    "local_cess": null,
    "health_cess": null,
    "education_cess": null,
    "total": null
  },
  "place": null,
  "date": null,
  "amount": null
}

Column meaning (left to right) in the holders table:
- "serial_number"       -> ಕ್ರಮ ಸಂಖ್ಯೆ  (the running row number 1, 2, 3 ...)
- "survey_hissa_number" -> ಸರ್ವೆ ಮತ್ತು ಹಿಸ್ಸಾ ನಂ. This is an ASCII code made
                           of digits, asterisks and forward slashes, e.g.
                           "8/*/*", "28/*/10", "1/*/3". Read it from the IMAGE
                           (it is printed clearly), NOT from the OCR text.
                           The separators are always forward slashes "/";
                           normalise any OCR artifact ("|", "I", spaces) back
                           to the "N/*/N" form (e.g. "28/*|10" -> "28/*/10",
                           "57/* I*" -> "57/*/*").
- "holder_name" +
  "father_name"         -> ಸ್ವಾದೀನದಾರರ ಮತ್ತು ತಂದೆಯ ಹೆಸರು
                           (holder name, and father/relation name if written)
- "extent"              -> ವಿಸ್ತೀರ್ಣ  (e.g. "0.22.00.00")
- "akara"               -> ಆಕಾರ
- "local_cess"          -> ಸ್ಥಳೀಯ ಕರ
- "health_cess"         -> ಅರೋಗ್ಯ ಕರ
- "education_cess"      -> ವಿದ್ಯಾ ಕರ
- "total"               -> ಒಟ್ಟು  (the rightmost total for that row)

Footer:
- "place"  -> ಮೊಕ್ಕಾಂ
- "date"   -> ದಿನಾಂಕ
- "amount" -> ಮೊತ್ತ

Rules:
- Preserve Kannada text exactly in Kannada Unicode. Do NOT transliterate.
- Create exactly one entry object per data row in the table, in order.
- Do NOT include the total ("ಒಟ್ಟು") summary row as an entry. Put it in
  "total_row" instead.
- "holder_name" is the person's name. If a relation word such as
  ಬಿನ್ / ಕೋಂ / ಉರುಪ್ / ತಂದೆ separates a second name, put that second name in
  "father_name"; otherwise leave "father_name" as null.
- Keep the dotted extent format EXACTLY as printed (e.g. "0.22.00.00",
  "2.04.08.00"). Do not reformat it.
- Read digits carefully. Do not copy column headings as values.

Numbers and zeros (extent, akara, local_cess, health_cess, education_cess,
total, and survey_hissa_number):
- Read ALL of these from the IMAGE. They are printed clearly; do not rely on
  the OCR text for them.
- Every DATA row shows a printed figure in all five money columns and in total,
  but that figure is FREQUENTLY 0. "Has a value" / "filled" does NOT mean
  "nonzero" — a printed 0 IS the value. Read what is printed, digit by digit;
  do not output null for a money cell of a data row.
- CRITICAL — do NOT FABRICATE numbers, and do NOT "correct" a printed 0. The
  #1 error on these documents is the model overriding legible zeros with the
  value it EXPECTS a tax column to hold. The cess columns usually contain
  numbers, so the model computes a "plausible" akara/cess/total from the extent
  or copies a neighbour — even when the cell plainly prints 0. Do not do this.
  Report ONLY the digits actually printed. If a cell prints 0, output "0".
- WHOLE-ROW ZERO rows are common and legitimate — NOT only for government /
  ಸರ್ಕಾರ / BDA / KIADB / ಭೂಸ್ವಾಧೀನ. A row for an ordinary PERSON very often
  shows 0 in ALL five money columns AND total (disputed / court-stay / suspended
  khata). These zeros are usually printed crisply and clearly. When akara..total
  all read 0, output "0" for every one — never substitute expected numbers.
  A wrong-but-plausible number is far worse than an honest 0.
- A grey diagonal "View Only" watermark crosses the table on every page. It is
  translucent — read the digit printed under it. If it genuinely hides a cell,
  do NOT guess; use the surrounding row (an otherwise all-zero row is 0).
- Keep the columns aligned. Because every money cell is filled, do NOT shift a
  value left into an empty-looking neighbour. If a row shows
  "0.03 0.02 0 0 0.05", that is akara=0.03, local_cess=0.02, health_cess=0,
  education_cess=0, total=0.05 — not akara=null, local_cess=0.03, ...
- Use null for a numeric field ONLY on the total summary row wording, never
  for a printed 0 in a data row.
- SELF-CHECK each data row: akara + local_cess + health_cess + education_cess
  is printed to equal total. If your reading of a row does not sum to its
  printed total, you mis-read or mis-aligned a cell — re-read that row from the
  image before answering.

CRITICAL - name and place fields (holder_name, father_name, district, taluk,
hobli, village):
- The IMAGE is your PRIMARY and authoritative source for these fields. This
  document is usually a crisp, digitally generated PDF, so the printed Kannada
  in the image is clean and complete. Read each name from the image.
- An OCR TEXT DUMP of this exact document is provided below the image. Treat it
  as a NOISY cross-check, NOT the source of truth. This pipeline's OCR
  frequently drops a leading initial (ವಿ, ಬಿ, ಬಿ ಡಿ), breaks a conjunct
  (ಭಟ್ಟ -> "ಭಲ ಲ"), or splits one name across lines.
- Reconcile the image against the OCR per name:
  (a) They agree -> use it.
  (b) The image is clearly legible and the OCR differs only by a dropped
      initial, a broken/garbled conjunct, or a split fragment -> TRUST THE
      IMAGE; do NOT copy the OCR's damage. NEVER drop a leading initial or
      letter that the image shows (image "ಬಿ.ವಿ.ಸಂಪತ್" beats OCR
      "ಬಿ.ಯಿಸಂಪತ್"; image "ವಿ.ಭಾಗ್ಯ" beats OCR "ಭಾಗ್ಯ"; image
      "ಬಿ ಡಿ ಎ ಗೆ ಭೂಸ್ವಾಧೀನ" beats OCR "ಎ ಗೆ ಭೂಸ್ವಾಧೀನ").
  (c) The image glyphs are genuinely blurry/ambiguous AND the OCR shows a
      clear, plausible Kannada name -> prefer the OCR reading.
  (d) NEVER replace a name with a different, more common-sounding name that
      NEITHER source supports (e.g. do not output ಬೋರೆಗೌಡ for ವೇಣುಗೋಪಾಲ).
- If a name is printed in Latin/English letters (e.g. "SHOBHA K.J.",
  "BALAJI PERNI VENKAIAH", "B.W.S.S.B"), keep it VERBATIM in Latin script. Do
  NOT transliterate it into Kannada and do NOT translate it.
- Assign each name to the row it occupies in the IMAGE. Do not let a stray OCR
  fragment that belongs to one row leak into an adjacent row's holder_name or
  father_name.
- When a name is split across short fragments/lines, join them into one name in
  reading order, but discard obvious OCR noise: a lone stray vowel or single
  letter wedged between real syllables (the "ಅ" in "ತಿಮ" / "ಅ" / "ಕ್ಕ") is
  noise — drop it, do not produce ತಿಮಅಕ್ಕ. You MAY restore an obviously
  dropped ottu/conjunct within that SAME name when the image supports it
  (ತಿಮ + ಕ್ಕ -> ತಿಮ್ಮಕ್ಕ). Repairing damage to the same name is allowed;
  swapping in an unrelated name is not.
"""


OCR_GROUNDING_HEADER = (
    "OCR TEXT DUMP of this exact document. This is a NOISY cross-check only —\n"
    "the IMAGE is authoritative for names, row alignment, and numbers. Use this\n"
    "text to catch wholesale name hallucination, not to override clear image\n"
    "glyphs (see the name-reconciliation rules above):\n"
    "--- BEGIN OCR TEXT ---\n"
)

OCR_GROUNDING_FOOTER = "\n--- END OCR TEXT ---\n"


# Tried in order. The Gemini free tier enforces a per-model daily request
# quota, so when the primary model is exhausted (429 RESOURCE_EXHAUSTED) the
# next model has its own separate quota bucket. This mirrors the PAN / Aadhaar
# extractors and is the fix for the per-model daily-quota flakiness.
#
# ACCURACY-FIRST ORDER (2026-07-16), matching the MR extractor: lead with the
# strongest reader, then fall through cheaper flash models. Every model here was
# verified callable on the project API key — the API's model-list ALSO returns
# models that 404 for this key (gemini-2.5-pro, gemini-3-pro-preview and the
# gemini-2.0-flash* pair were all NOT_FOUND), so do NOT add one without
# confirming it answers. The old list led with gemini-2.5-flash and ended with
# the two dead 2.0-flash models, which burned retries on every Patta doc.
# VERIFIED CALLABLE on this project key 2026-08-31. The previous list was
# entirely / partly RETIRED — Google returns 404 "no longer available to new
# users" for every gemini-2.5-* and gemini-2.0-* name, so those entries could
# never answer and each burned 3 retries before the next was tried. Google's own
# migration targets: 2.5-flash and 2.0-flash -> 3.6-flash, 2.5-flash-lite and
# 2.0-flash-lite -> 3.5-flash-lite. Do NOT add a name without confirming it
# answers on this key; models.list() also returns names that 404.
PATTA_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def _is_quota_error(error) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _is_missing_model_error(error) -> bool:
    """A 404 NOT_FOUND (model unavailable to this key / misspelled) is permanent,
    not transient — retrying it just burns the backoff budget on a dead name."""
    message = str(error)
    return "404" in message or "NOT_FOUND" in message


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
    if value is None:
        return None

    value = re.sub(r"\s+", " ", str(value)).strip()

    return value or None


def _parse_decimal(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _row_numeric_status(entry: dict) -> str:
    """
    In these records each DATA row is printed so that
    akara + local_cess + health_cess + education_cess == total.

    Use that as an integrity check to flag rows whose numeric cells were
    mis-read or column-shifted, so they can be routed to manual review.
    Returns "ok", "mismatch", or "unknown" (a cell could not be parsed).
    """
    parts = [
        _parse_decimal(entry.get("akara")),
        _parse_decimal(entry.get("local_cess")),
        _parse_decimal(entry.get("health_cess")),
        _parse_decimal(entry.get("education_cess")),
    ]
    total = _parse_decimal(entry.get("total"))

    if total is None or any(part is None for part in parts):
        return "unknown"

    # Tolerance absorbs the documents' own rounding (each cess is rounded to
    # 2 dp before being summed into the printed total).
    return "ok" if abs(sum(parts) - total) <= 0.02 else "mismatch"


# The money columns a per-entry row and the printed grand-total row share.
_TOTAL_COLUMNS = ("akara", "local_cess", "health_cess", "education_cess", "total")


def _reconcile_totals(entries: list, total_row: dict) -> dict:
    """
    Document-level integrity check. The printed ಒಟ್ಟು grand-total row equals the
    column-wise sum of every data row. Compare each money column's entry-sum
    against the printed grand total.

    This catches errors the per-row `numeric_check` is BLIND to — most importantly
    FABRICATED rows that are internally self-consistent (e.g. an all-zero row read
    as 0.7/0.35/0.1/0.07/1.22 sums fine on its own row but inflates the column
    grand totals, so it fails here).

    Returns per-column "ok"/"mismatch"/"unknown" plus an overall status.

    Tolerance is a tight FLAT value, not row-count-scaled: the printed grand
    total is an exact sum of 2-decimal cells, so a correct extraction reconciles
    to the cent. A small flat slack absorbs a stray 0.01 rounding oddity while
    still catching a single fabricated or misread row.
    """
    columns = {}
    tolerance = 0.10

    for column in _TOTAL_COLUMNS:
        printed = _parse_decimal(total_row.get(column))
        parts = [_parse_decimal(entry.get(column)) for entry in entries]

        if printed is None or any(part is None for part in parts):
            columns[column] = "unknown"
            continue

        columns[column] = (
            "ok" if abs(sum(parts) - printed) <= tolerance else "mismatch"
        )

    if any(status == "mismatch" for status in columns.values()):
        overall = "mismatch"
    elif all(status == "ok" for status in columns.values()):
        overall = "ok"
    else:
        overall = "unknown"

    return {"status": overall, "columns": columns}


def _normalize_entry(entry: dict) -> dict:
    normalized = {
        "serial_number": _clean_str(entry.get("serial_number")),
        "survey_hissa_number": _clean_str(entry.get("survey_hissa_number")),
        "holder_name": _clean_str(entry.get("holder_name")),
        "father_name": _clean_str(entry.get("father_name")),
        "extent": _clean_str(entry.get("extent")),
        "akara": _clean_str(entry.get("akara")),
        "local_cess": _clean_str(entry.get("local_cess")),
        "health_cess": _clean_str(entry.get("health_cess")),
        "education_cess": _clean_str(entry.get("education_cess")),
        "total": _clean_str(entry.get("total")),
    }
    normalized["numeric_check"] = _row_numeric_status(normalized)
    return normalized


def _normalize_total_row(total_row) -> dict:
    total_row = total_row if isinstance(total_row, dict) else {}

    return {
        "extent": _clean_str(total_row.get("extent")),
        "akara": _clean_str(total_row.get("akara")),
        "local_cess": _clean_str(total_row.get("local_cess")),
        "health_cess": _clean_str(total_row.get("health_cess")),
        "education_cess": _clean_str(total_row.get("education_cess")),
        "total": _clean_str(total_row.get("total")),
    }


def _fold_text(value) -> str:
    """
    Fold a name/survey value for COMPARISON only (not for output). Drops spaces,
    dots and case so that trivial formatting differences between two model runs
    ("ಡಿ.ಬಿ. ಸುರೇಶ್" vs "ಡಿಬಿ ಸುರೇಶ್", "16/*|1" vs "16/*/1") do NOT count as a
    disagreement, while a real character difference (ಟಿ vs ಲಿ, a dropped ಸ್ವಾಮಿ,
    2B vs 23) still does.
    """
    if value is None:
        return ""
    # Normalise survey separators ("|", "\" -> "/") so OCR-ish variants compare
    # equal, THEN drop spaces, dots and zero-width joiners (formatting noise).
    folded = str(value).replace("\\", "/").replace("|", "/")
    folded = re.sub(r"[\s.‌]", "", folded)
    return folded.casefold()


_TEXT_COMPARE_FIELDS = ("holder_name", "father_name", "survey_hissa_number")


def _diff_text_fields(primary_entries: list, secondary_entries: list) -> dict:
    """
    Compare two independent extractions row-by-row (aligned on serial_number) on
    the text fields that have no arithmetic check — holder_name, father_name and
    survey_hissa_number. Rows where the two runs disagree are surfaced for review.

    This is the text analogue of the numeric checksum: it cannot PROVE a value is
    right, but a disagreement between two independent reads is a strong signal that
    a cell needs a human look. LIMITATION: correlated errors (both models make the
    same glyph mistake) are invisible here, so it catches a subset, not all.

    Pure function (no I/O) so the alignment/diff logic is unit-testable.
    """
    secondary_by_serial = {
        _clean_str(entry.get("serial_number")): entry
        for entry in secondary_entries
        if isinstance(entry, dict)
    }

    disagreement_rows = []
    details = {}
    rows_compared = 0

    for entry in primary_entries:
        serial = _clean_str(entry.get("serial_number"))
        other = secondary_by_serial.get(serial)

        # A row the verifier didn't produce is NOT evidence the primary's text is
        # wrong — it usually means the verifier under-extracted (common on long
        # docs / weaker models). Skip it; caller judges overall coverage instead.
        if other is None:
            continue

        rows_compared += 1
        differing = [
            field
            for field in _TEXT_COMPARE_FIELDS
            if _fold_text(entry.get(field)) != _fold_text(other.get(field))
        ]
        if differing:
            disagreement_rows.append(serial)
            details[serial] = differing

    return {
        "disagreement_rows": disagreement_rows,
        "details": details,
        "row_count_primary": len(primary_entries),
        "row_count_secondary": len(secondary_entries),
        "rows_compared": rows_compared,
    }


def _try_model(client, contents, model):
    """
    Call a SINGLE model, retrying transient failures with exponential backoff.
    A daily-quota (429) is NOT transient, so we stop early. Returns
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
            print(f"Gemini PATTA {model} attempt {attempt + 1} failed:", e)
            if _is_quota_error(e) or _is_missing_model_error(e):
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

        if response and getattr(response, "text", None):
            return response.text, last_error

    return None, last_error


def _run_patta_models(client, contents, models):
    """
    Try each model in order until one returns usable text. Returns
    (text, used_model, last_error); text/used_model are None if all failed.
    """
    last_error = None
    for model in models:
        text, error = _try_model(client, contents, model)
        if error is not None:
            last_error = error
        if text:
            print(f"Gemini PATTA extracted using model: {model}")
            return text, model, last_error
    return None, None, last_error


def _extract_once(client, contents, model):
    """
    Run ONE model and score its output. Returns (result_dict_or_None, last_error).
    The result carries the parsed entries plus the integrity verdicts used to
    decide whether this read can be trusted or should trigger a repair retry.
    """
    text, error = _try_model(client, contents, model)
    if not text:
        return None, error

    data = _extract_json(text)
    if not isinstance(data, dict):
        return None, error

    raw_entries = data.get("entries")
    entries = [
        _normalize_entry(entry)
        for entry in raw_entries
        if isinstance(entry, dict)
    ] if isinstance(raw_entries, list) else []

    total_row = _normalize_total_row(data.get("total_row"))
    numeric_mismatch_rows = [
        entry.get("serial_number")
        for entry in entries
        if entry.get("numeric_check") == "mismatch"
    ]
    reconciliation = _reconcile_totals(entries, total_row)

    # A read is fully trustworthy ONLY when its entry columns actually sum to the
    # printed grand total (reconciliation "ok"). This proves it is BOTH complete
    # (no missing rows — a truncated read can't sum to the printed total) AND
    # correct (no wrong numbers). "unknown" (no total row parsed) is NOT trusted:
    # it usually means the read stopped early, which must never look "clean".
    numbers_trustworthy = (
        not numeric_mismatch_rows and reconciliation["status"] == "ok"
    )

    result = {
        "model": model,
        "data": data,
        "entries": entries,
        "total_row": total_row,
        "numeric_mismatch_rows": numeric_mismatch_rows,
        "reconciliation": reconciliation,
        "numbers_trustworthy": numbers_trustworthy,
    }
    return result, error


def _rank_result(result):
    """
    Sort key for choosing the best read (lower is better). Ordered so a COMPLETE
    read always beats a truncated one:
      1. reconciliation tier: ok (0) < mismatch (1) < unknown/no-total-row (2).
         'unknown' is worst because a read with no parseable grand-total row is
         almost always truncated — it must not win by looking "clean".
      2. fewer per-row + column problems.
      3. more rows (a complete read outranks a short one at the same tier).
    """
    status = result["reconciliation"]["status"]
    tier = {"ok": 0, "mismatch": 1}.get(status, 2)
    return (tier, _result_problem_score(result), -len(result["entries"]))


def _result_problem_score(result) -> int:
    """
    Fewer problems is better: per-row checksum failures plus grand-total columns
    that don't reconcile. Used as a tie-breaker within a reconciliation tier.
    """
    recon_columns = result["reconciliation"].get("columns", {})
    recon_bad = sum(1 for status in recon_columns.values() if status == "mismatch")
    return len(result["numeric_mismatch_rows"]) + recon_bad


_MONEY_FIELDS = ("akara", "local_cess", "health_cess", "education_cess", "total")


def _repair_numeric_rows(best_entries, all_results):
    """
    Assemble cleaner rows across attempts. For any row in `best_entries` that
    fails its own checksum (numeric_check == "mismatch"), find the SAME serial in
    another attempt whose money cells self-consistently sum (numeric_check ==
    "ok") and adopt that attempt's NUMBERS for the row (text fields are kept from
    the accepted read). Each row has an independent checksum, so this fixes the
    scattered stochastic misreads that no single whole read got right.

    NOTE: cannot fix a FABRICATED all-zero row — a fabricated row is itself
    self-consistent (numeric_check == "ok"), so it is never a mismatch here;
    fabrication is handled by the whole-read reconciliation selection instead.

    Returns (repaired_entries, repaired_serials). Pure function.
    """
    ok_numbers_by_serial = {}
    for result in all_results:
        for entry in result["entries"]:
            serial = entry.get("serial_number")
            if entry.get("numeric_check") == "ok" and serial not in ok_numbers_by_serial:
                ok_numbers_by_serial[serial] = entry

    repaired_entries = []
    repaired_serials = []

    for entry in best_entries:
        serial = entry.get("serial_number")
        donor = ok_numbers_by_serial.get(serial)
        if entry.get("numeric_check") == "mismatch" and donor is not None:
            fixed = dict(entry)
            for field in _MONEY_FIELDS:
                fixed[field] = donor.get(field)
            fixed["numeric_check"] = _row_numeric_status(fixed)
            repaired_entries.append(fixed)
            repaired_serials.append(serial)
        else:
            repaired_entries.append(entry)

    return repaired_entries, repaired_serials


_MAX_GUIDED_COMBINATIONS = 4096


def _money_key(entry):
    """Parsed (akara, local, health, edu, total) tuple, or None if any unparseable."""
    key = tuple(_parse_decimal(entry.get(field)) for field in _MONEY_FIELDS)
    return None if any(value is None for value in key) else key


def _reconcile_guided_repair(best_entries, all_results, total_row):
    """
    Use the printed grand-total row as ground truth to fix rows the checksums
    CAN'T — most importantly FABRICATED all-zero rows, which are self-consistent
    (so `numeric_check` == ok) but inflate the column totals.

    Across attempts, a fabricated row reads differently (e.g. one attempt
    "0 0 0 0 0", another "0.91 0.45 0.14 0.09 1.59"). Only ONE of those readings
    lets the column sums equal the printed grand total. So for every row whose
    money cells DISAGREE across attempts, we choose the real reading (never an
    invented value) that drives the document's column sums to the printed totals.

    Rows all attempts agree on are fixed and never touched. Search is bounded:
    enumerate if the combination count is small, else greedy-descend. Only real
    per-attempt readings are candidates, so this never fabricates.

    Returns (repaired_entries, repaired_serials). Pure function.
    """
    printed = {field: _parse_decimal(total_row.get(field)) for field in _MONEY_FIELDS}
    if any(value is None for value in printed.values()):
        return best_entries, []  # no grand total to steer by

    # Distinct money-readings per serial across every attempt.
    candidates_by_serial = {}
    for result in all_results:
        for entry in result["entries"]:
            key = _money_key(entry)
            if key is None:
                continue
            serial = entry.get("serial_number")
            candidates_by_serial.setdefault(serial, {}).setdefault(key, entry)

    # A row is "variable" only if attempts disagree on its numbers.
    variable = []  # (index, list-of-candidate-entries)
    for index, entry in enumerate(best_entries):
        serial = entry.get("serial_number")
        options = candidates_by_serial.get(serial)
        if _money_key(entry) is not None and options and len(options) > 1:
            variable.append((index, list(options.values())))

    if not variable:
        return best_entries, []

    # Column sums contributed by the fixed (non-variable) rows.
    variable_indices = {index for index, _ in variable}
    fixed_sum = {field: 0.0 for field in _MONEY_FIELDS}
    for index, entry in enumerate(best_entries):
        if index in variable_indices:
            continue
        key = _money_key(entry)
        if key is None:
            continue
        for field, value in zip(_MONEY_FIELDS, key):
            fixed_sum[field] += value

    def deviation(choice_by_index):
        sums = dict(fixed_sum)
        for index, chosen in choice_by_index.items():
            for field, value in zip(_MONEY_FIELDS, _money_key(chosen)):
                sums[field] += value
        return sum(abs(sums[f] - printed[f]) for f in _MONEY_FIELDS)

    # Start from best's own reading for each variable row.
    def best_option(index, options):
        target = _money_key(best_entries[index])
        return next((o for o in options if _money_key(o) == target), options[0])

    choice = {index: best_option(index, options) for index, options in variable}

    combo_count = 1
    for _, options in variable:
        combo_count *= len(options)

    if combo_count <= _MAX_GUIDED_COMBINATIONS:
        best_dev, best_choice = None, choice
        for combo in itertools.product(*[options for _, options in variable]):
            trial = {variable[k][0]: combo[k] for k in range(len(variable))}
            dev = deviation(trial)
            if best_dev is None or dev < best_dev:
                best_dev, best_choice = dev, trial
        choice = best_choice
    else:
        # Too many combinations — greedy coordinate descent.
        improved, passes = True, 0
        while improved and passes < 3:
            improved, passes = False, passes + 1
            for index, options in variable:
                current = deviation(choice)
                for option in options:
                    trial = dict(choice)
                    trial[index] = option
                    if deviation(trial) < current:
                        choice[index] = option
                        current = deviation(trial)
                        improved = True

    repaired_entries = []
    repaired_serials = []
    for index, entry in enumerate(best_entries):
        chosen = choice.get(index)
        if chosen is not None and _money_key(chosen) != _money_key(entry):
            fixed = dict(entry)
            for field in _MONEY_FIELDS:
                fixed[field] = chosen.get(field)
            fixed["numeric_check"] = _row_numeric_status(fixed)
            repaired_entries.append(fixed)
            repaired_serials.append(entry.get("serial_number"))
        else:
            repaired_entries.append(entry)

    return repaired_entries, repaired_serials


# The header keys a Gemini read emits. The OCR fallback fills the same set (null
# where it could not recover a value) so both paths return one identical shape.
_HEADER_OUTPUT_FIELDS = ("khata_number", "district", "taluk", "hobli", "village")


def _unavailable_result(error, ocr_text: str = None) -> dict:
    """
    Shape returned when NO model produced a read (e.g. an exhausted API quota).

    Falls back to the OCR text for the LABELLED HEADER, the footer, the grand-
    total row and the survey codes, which it prints reliably. The holder table is
    deliberately NOT reconstructed: the Kannada names fuse/split in this OCR, and
    a silently-wrong owner in a title record is far worse than an absent one. So
    "entries" stays empty and is flagged for manual review, and the reason for the
    degraded read is reported explicitly.

    Emits the SAME key set a Gemini read produces (verification keys carry neutral
    values), so callers never branch on which path ran.
    """
    quota = error is not None and _is_quota_error(error)
    reason = "quota_exhausted" if quota else "model_unavailable"

    recovered = {}
    try:
        recovered = extract_patta_book_from_ocr(ocr_text) or {}
    except Exception as fallback_error:  # never let the fallback mask the outage
        print("Patta OCR fallback failed:", fallback_error)

    surveys = recovered.pop("affected_survey_numbers", [])
    total_row = _normalize_total_row(recovered.get("total_row"))
    header_recovered = any(recovered.get(field) for field in _HEADER_OUTPUT_FIELDS)

    result = {
        "document_type": "PATTA_BOOK",
        **{field: recovered.get(field) for field in _HEADER_OUTPUT_FIELDS},
        "entries": [],
        "total_row": total_row,
        "affected_survey_numbers": surveys,
        "place": recovered.get("place"),
        "date": recovered.get("date"),
        "amount": recovered.get("amount"),
        # No model answered, so no numeric/text checks ran. Present the same
        # verification keys a Gemini read carries, with neutral values.
        "numeric_mismatch_rows": [],
        "totals_reconciliation": {"status": "unavailable", "columns": {}},
        "extraction_attempts": 0,
        "repaired_rows": [],
        "text_verification": {"status": "unavailable", "verified_by": None},
        "text_disagreement_rows": [],
        "field_status": {
            "entries": "requires_manual_review",
            "entries_numeric": "requires_manual_review",
            "entries_text": "requires_manual_review",
            **{field: "ocr_fallback_review" for field in _HEADER_OUTPUT_FIELDS
               if recovered.get(field)},
        },
        "extraction_status": (
            "partial_requires_review" if header_recovered else "unavailable"
        ),
        "unavailable_reason": reason,
        "message": (
            (
                "Gemini API quota exhausted — no model could read the document. "
                "Retry once the quota resets or use a billed API key."
                if quota else
                "No Gemini model returned a usable read for this document."
            )
            + (
                " Header/footer fields below were recovered by OCR and are LOW"
                " CONFIDENCE; the holder table was NOT reconstructed (names are"
                " not recoverable from this OCR). Verify every value before use."
                if header_recovered else ""
            )
        ),
        "error": str(error) if error is not None else None,
        "extraction_source": (
            "ocr_header_fallback_v1" if header_recovered
            else "gemini_vision_patta_v1:unavailable"
        ),
    }
    return result


def extract_patta_book_gemini(file_path: str, ocr_text: str = None):
    """
    Extract Karnataka Khata / Patta Book fields from the document using Gemini
    vision, grounded by the pipeline's OCR text.

    Gemini vision reads the table structure and numeric columns very well, but
    it hallucinates Kannada names (it "autocorrects" them to common-sounding
    names). The OCR text usually has the correct glyphs, so we pass it in as
    the authoritative source for Kannada names while the image drives row
    alignment and number reading.

    Returns a structured dict. When Gemini is unavailable (no key, exhausted
    quota, or a failed image load) it returns the OCR header/footer fallback
    shape instead of None, so the caller always has the labelled fields plus an
    explicit reason for the degraded read.
    """
    if not settings.GOOGLE_API_KEY:
        return _unavailable_result(None, ocr_text)

    try:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        image = _load_image(file_path)
    except Exception as setup_error:
        print("Gemini PATTA setup failed (client/image):", setup_error)
        return _unavailable_result(setup_error, ocr_text)

    # Grounding text is optional. When available it is the single biggest
    # accuracy lever for the Kannada name fields.
    contents = [PATTA_PROMPT]

    if ocr_text and ocr_text.strip():
        contents.append(
            OCR_GROUNDING_HEADER + ocr_text.strip() + OCR_GROUNDING_FOOTER
        )

    contents.append(image)

    # Self-repair loop: Gemini at temp=0 is non-deterministic, so a doc can
    # fabricate numbers on one call and read them correctly on the next. Try
    # models in order; accept the FIRST read whose entry columns reconcile to the
    # printed grand total. Keep every successful read so a non-accepted one can
    # double as the text-verification sample (no extra call). Availability
    # fallback is preserved: a model that fails entirely is skipped for free.
    max_attempts = (
        settings.PATTA_RECONCILE_MAX_ATTEMPTS
        if settings.PATTA_RECONCILE_RETRY else 1
    )

    best = None
    successful_results = []
    last_error = None

    for model in PATTA_MODELS:
        result, error = _extract_once(client, contents, model)
        if error is not None:
            last_error = error
        if result is None:
            continue  # model unavailable/unparseable — try the next, no cost to budget

        successful_results.append(result)

        # Keep the best-ranked read: a COMPLETE reconciling read beats a truncated
        # one that merely looks clean (see _rank_result).
        if best is None or _rank_result(result) < _rank_result(best):
            best = result

        if best["numbers_trustworthy"]:
            break  # complete and reconciles — accept it
        if len(successful_results) >= max_attempts:
            break  # spent the repair budget; keep the best read so far

    if best is None:
        if last_error:
            print("Gemini PATTA extraction giving up after all models:", last_error)
        # No model produced a read (e.g. an exhausted API quota). Degrade to the
        # OCR header/footer fallback instead of returning None, and report WHY —
        # a quota outage is otherwise indistinguishable from a genuine failure.
        return _unavailable_result(last_error, ocr_text)

    used_model = best["model"]
    data = best["data"]
    total_row = best["total_row"]
    extraction_attempts = len(successful_results)

    # Two-stage numeric repair across the attempts, only when the accepted read
    # doesn't already reconcile:
    #   (1) row-level — fix checksum-FAILING rows from a self-consistent version.
    #   (2) reconciliation-guided — fix checksum-PASSING but fabricated rows
    #       (e.g. an all-zero row read as nonzero) using the printed grand total.
    entries = best["entries"]
    repaired_rows = []
    if len(successful_results) > 1 and not best["numbers_trustworthy"]:
        entries, repaired_rows = _repair_numeric_rows(entries, successful_results)

        totals_reconciliation = _reconcile_totals(entries, total_row)
        if totals_reconciliation["status"] == "mismatch":
            entries, guided_rows = _reconcile_guided_repair(
                entries, successful_results, total_row
            )
            repaired_rows = sorted(set(repaired_rows) | set(guided_rows), key=str)

    numeric_mismatch_rows = [
        entry.get("serial_number")
        for entry in entries
        if entry.get("numeric_check") == "mismatch"
    ]
    totals_reconciliation = _reconcile_totals(entries, total_row)
    numbers_trustworthy = (
        not numeric_mismatch_rows and totals_reconciliation["status"] == "ok"
    )

    # Text verification: flag rows where two independent reads disagree on the
    # un-checkable text fields (holder/father/survey). Reuse a different-model read
    # from the repair loop when one is COMPARABLE (similar row count) — a read that
    # under-extracted is useless as a verifier and only produces noise. Only spend
    # an extra call if no comparable spare exists. Best-effort — never blocks.
    text_verification = {"status": "disabled"}
    text_disagreement_rows = []

    if settings.PATTA_VERIFY_TEXT_FIELDS and entries:
        target_rows = len(entries)

        def _is_comparable(candidate_entries):
            # Require the candidate to cover most of the primary's rows, else the
            # comparison is dominated by the verifier's own truncation.
            return target_rows and len(candidate_entries) >= 0.8 * target_rows

        spares = [
            {"model": r["model"], "entries": r["entries"]}
            for r in successful_results
            if r["model"] != used_model and _is_comparable(r["entries"])
        ]
        other = spares[0] if spares else None

        if other is None:
            verify_models = [model for model in PATTA_MODELS if model != used_model]
            verify_text, verify_model, _ = _run_patta_models(client, contents, verify_models)
            if verify_text:
                verify_data = _extract_json(verify_text)
                other = {
                    "model": verify_model,
                    "entries": [
                        _normalize_entry(entry)
                        for entry in verify_data.get("entries", [])
                        if isinstance(entry, dict)
                    ] if isinstance(verify_data, dict) else [],
                }

        if other is not None:
            diff = _diff_text_fields(entries, other["entries"])
            coverage = diff["rows_compared"] / target_rows if target_rows else 0

            if coverage < 0.6:
                # The verifier reached too few rows to be meaningful — report as
                # inconclusive and do NOT flag rows (avoids row_missing noise).
                text_verification = {
                    "status": "inconclusive",
                    "verified_by": other["model"],
                    "rows_compared": diff["rows_compared"],
                    "row_count": {
                        "primary": diff["row_count_primary"],
                        "secondary": diff["row_count_secondary"],
                    },
                }
            else:
                text_disagreement_rows = diff["disagreement_rows"]
                text_verification = {
                    "status": "mismatch" if text_disagreement_rows else "ok",
                    "verified_by": other["model"],
                    "details": diff["details"],
                    "rows_compared": diff["rows_compared"],
                    "row_count": {
                        "primary": diff["row_count_primary"],
                        "secondary": diff["row_count_secondary"],
                    },
                }
        else:
            text_verification = {"status": "unavailable", "verified_by": None}

    khata_number = _clean_str(data.get("khata_number"))
    district = _clean_str(data.get("district"))
    taluk = _clean_str(data.get("taluk"))
    hobli = _clean_str(data.get("hobli"))
    village = _clean_str(data.get("village"))

    header_present = any([khata_number, taluk, hobli, village])

    # Text is trustworthy unless the verification pass ran AND found a disagreement.
    # "disabled"/"unavailable" verification does not by itself force review.
    text_trustworthy = not text_disagreement_rows

    return {
        "document_type": "PATTA_BOOK",
        "khata_number": khata_number,
        "district": district,
        "taluk": taluk,
        "hobli": hobli,
        "village": village,
        "entries": entries,
        "total_row": total_row,
        "place": _clean_str(data.get("place")),
        "date": _clean_str(data.get("date")),
        "amount": _clean_str(data.get("amount")),
        "numeric_mismatch_rows": numeric_mismatch_rows,
        "totals_reconciliation": totals_reconciliation,
        "extraction_attempts": extraction_attempts,
        "repaired_rows": repaired_rows,
        "text_verification": text_verification,
        "text_disagreement_rows": text_disagreement_rows,
        "field_status": {
            "khata_number": "extracted" if khata_number else "requires_manual_review",
            "taluk": "extracted" if taluk else "requires_manual_review",
            "hobli": "extracted" if hobli else "requires_manual_review",
            "village": "extracted" if village else "requires_manual_review",
            "entries": "extracted" if entries else "requires_manual_review",
            "entries_numeric": (
                "extracted" if entries and numbers_trustworthy
                else "requires_manual_review"
            ),
            "entries_text": (
                "extracted" if entries and text_trustworthy
                else "requires_manual_review"
            ),
        },
        "extraction_status": (
            "extracted"
            if header_present and entries and numbers_trustworthy and text_trustworthy
            else "partial_requires_review"
        ),
        "extraction_source": f"gemini_vision_patta_v1:{used_model}",
    }

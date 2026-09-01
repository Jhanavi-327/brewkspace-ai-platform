import json
import re
import time

import fitz
from PIL import Image
from google import genai
from google.genai import types

from app.core.config import settings
from app.services.mr.mr_ocr_fallback import extract_mutation_register_from_ocr
from app.services.mr.mr_table_ocr import extract_mr_table_from_image


# Deterministic decoding: this is an extraction task, so we want the same
# document to yield the same output on every run (no sampling variance).
GENERATION_CONFIG = types.GenerateContentConfig(temperature=0)


MR_PROMPT = """
You are extracting fields from a Karnataka "Mutation Register copy"
(ನಮೂನೆ-12 / ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್ ಪ್ರತಿ, ನಿಯಮ 46). This is the government
record of a single mutation (ಮ್ಯುಟೇಶನ್ / ವಹಿವಾಟು) on a survey number: a header
of key-value fields, a status stamp, one transaction table, a free-text order
paragraph, and a footer with the order place/date and the signing officer.

Return ONLY valid JSON. No markdown. No explanation.

The document may be supplied as SEVERAL page images (the holder table can
continue across pages). Read ALL of them and extract every table row on every
page, in order, into a single "entries" array.

Extract exactly this structure:

{
  "namune": null,
  "district": null,
  "taluk": null,
  "hobli": null,
  "village": null,
  "disposal_year": null,
  "disposal_number": null,
  "mr_number": null,
  "source": null,
  "source_number": null,
  "change_type": null,
  "possession_type": null,
  "order_reference_date": null,
  "status": null,
  "survey_details": {
    "survey_number": null,
    "total_extent": null,
    "kharab_a": null,
    "kharab_b": null,
    "revenue": null,
    "judi": null,
    "cess": null,
    "water_rate": null,
    "patta": null,
    "soil_type": null
  },
  "affected_survey_numbers": [],
  "entries": [
    {
      "survey_hissa_number": null,
      "patta_number": null,
      "revenue": null,
      "rights_changed_from": null,
      "extent_from": null,
      "rights_changed_to": null,
      "extent_to": null,
      "relation": null,
      "khata_number": null,
      "possession_mode": null,
      "rights": null,
      "liabilities": null,
      "row_status": null,
      "inspecting_officer": null
    }
  ],
  "mutation_order_details": null,
  "place": null,
  "order_date": null,
  "signatory": null
}

Read every value from the document itself. The mappings below name each field
by its printed Kannada label ONLY — they do NOT contain sample values, and you
must NOT invent, assume, or carry over any value; output exactly what THIS
document prints, or null.

Header fields (the key-value block at the top):
- "namune"              -> the ನಮೂನೆ form number printed in the title
- "district"            -> ಜಿಲ್ಲೆ
- "taluk"               -> ತಾಲ್ಲೂಕು
- "hobli"               -> ಹೋಬಳಿ
- "village"             -> ಗ್ರಾಮ
- "disposal_year"       -> ವಹಿವಾಟು ವರ್ಷ
- "disposal_number"     -> ವಹಿವಾಟು ಸಂಖ್ಯೆ — a short running number in the header.
                           The value often prints tight against its label with no
                           gap, or wraps onto the next line, so look for it
                           carefully in the IMAGE. It is DISTINCT from
                           "source_number" (the long ಸಂಖ್ಯೆ reference code at the
                           top-right). Leave null only when no such number is
                           printed at all.
- "mr_number"           -> M.R.ನಂಬರ್
- "source"              -> ಮೂಲ (the origin/authority for the mutation)
- "source_number"       -> ಸಂಖ್ಯೆ (the reference number at top-right). Read this
                           alphanumeric code from the IMAGE, character by
                           character; it is printed clearly. Do NOT rely on the
                           OCR text for it.
- "change_type"         -> ಬದಲಾವಣೆ ರೀತಿ
- "possession_type"     -> ಸ್ವಾಧೀನತೆ ರೀತಿ
- "order_reference_date"-> ದಿನಾಂಕ printed in the top-right header block (the
                           order's reference date). DISTINCT from the footer
                           "order_date".

Status stamp:
- "status" -> ONLY the standalone approval/rejection stamp printed between the
  header block and the transaction table (usually flanked by ***** asterisks):
  ಅನುಮೋದಿಸಿದೆ (approved) or ತಿರಸ್ಕರಿಸಿದೆ (rejected). Return just that Kannada
  word, without the asterisks. If no such stamp is printed between the header and
  the table, "status" MUST be null. Do NOT derive it from the footer
  ಮ್ಯುಟೇಶನ್ ಆದೇಶದ ವಿವರ order paragraph — a word like ಅಂಗೀಕರಿಸಿದೆ appearing in
  that paragraph is NOT the stamp and must NOT be put in "status".

Transaction table — the common layout always has these SIX columns left to right:
ಬಾಧಿತವಾದ ಸರ್ವೆ ಮತ್ತು ಹಿಸ್ಸಾ ನಂ. | ಹಕ್ಕು ಬದಲಾವಣೆ ಮಾಡಿದವರು | ವಿಸ್ತೀರ್ಣ |
ಹಕ್ಕು ಬದಲಾವಣೆ ಪಡೆದವರು | ವಿಸ್ತೀರ್ಣ | ತಪಾಸಣೆ ಅಧಿಕಾರಿ ರುಜು/ಷರಾ

READ THIS TABLE COLUMN-BY-COLUMN, NOT ROW-BY-ROW. Its cells do NOT reliably line
up into rows. Two shapes occur:
  (a) genuinely row-wise — each printed row has its own survey, party and extent;
  (b) all the survey numbers STACKED together in the first cell, while the party
      names sit on separate rows beside them.
COMPLETENESS IS THE PRIORITY:
- "affected_survey_numbers" -> a JSON array of EVERY survey/hissa number printed
  in the first column, in order, however they are stacked. Never omit one.
- EVERY name printed in ಹಕ್ಕು ಬದಲಾವಣೆ ಮಾಡಿದವರು and in ಹಕ್ಕು ಬದಲಾವಣೆ ಪಡೆದವರು must
  appear in the output. Never drop a name.
- Do NOT invent a pairing. Set an entry's "survey_hissa_number" ONLY when that
  survey is genuinely printed on that party's own row (shape (a)). In shape (b),
  leave each entry's "survey_hissa_number" null — the full list already lives in
  "affected_survey_numbers".

SURVEY / HISSA CODES ARE ASCII — NEVER KANNADA. This applies to every
"survey_hissa_number" and to "affected_survey_numbers":
- Copy the code EXACTLY as printed, in Latin/ASCII: digits, "*", "/", and a Latin
  hissa letter (P or p) when one is printed. Preserve its letter case.
- NEVER transliterate a Latin hissa letter into a Kannada letter (and never the
  reverse). A Kannada letter must not appear in these fields.
- Preserve EVERY "/"-separated segment, including a middle "*". Do NOT collapse,
  merge or drop a segment — a three-segment code stays three segments.
- The ಮ್ಯುಟೇಶನ್ ಆದೇಶದ ವಿವರ order paragraph often spells the same survey number in
  Kannada letters. IGNORE that spelling for these fields; read the code from the
  TABLE cell, in ASCII.

Per-entry fields:
- "survey_hissa_number" -> ಬಾಧಿತವಾದ ಸರ್ವೆ ಮತ್ತು ಹಿಸ್ಸಾ ನಂ. for THIS row, when the
                           row genuinely has its own (an ASCII code of digits,
                           asterisks and forward slashes). Read it from the IMAGE;
                           normalise OCR separators ("|", "I", spaces) back to
                           "/". Leave null rather than guessing.
- "patta_number"        -> the ಪಟ್ಟಾ NUMBER printed under the survey cell. It is a
                           number. Leave it null when only the label ಪಟ್ಟಾ (with
                           nothing after it) is printed. NEVER put a party name,
                           or any part of one, here.
- "revenue"             -> ಕಂದಾಯ value, when printed under the survey cell.

A PARTY NAME IS ONE WHOLE VALUE. Copy the entire name from its own cell into a
single name field. NEVER split a name across two fields, and never let the first
words of a name land in "patta_number", "revenue" or any other column.
- "rights_changed_from" -> ಹಕ್ಕು ಬದಲಾವಣೆ ಮಾಡಿದವರು (the party losing/transferring
                           the right — the "from" name).
- "extent_from"         -> ವಿಸ್ತೀರ್ಣ printed next to the "from" party.
- "rights_changed_to"   -> ಹಕ್ಕು ಬದಲಾವಣೆ ಪಡೆದವರು (the party gaining/receiving the
                           right — the "to" name).
- "extent_to"           -> ವಿಸ್ತೀರ್ಣ printed next to the "to" party.
- "inspecting_officer"  -> ತಪಾಸಣೆ ಅಧಿಕಾರಿ ರುಜು/ಪರಾ, when a name/value is printed.

A PARTY CELL IS A BLOCK, NOT JUST A NAME (very important). Under a party's name
the same cell often also prints:
  - "ಋಣಗಳು : [ ... MR ... ]" — that party's debts / charges / MR references
  - "ಸಂಬಂಧ : <x>"            — that party's relationship annotation
Everything in the block belongs to the SAME party as the name above it. Emit ONE
entry per PARTY, carrying that party's name, extent, liabilities and relation
TOGETHER in that single entry:
- "rights_changed_from" / "rights_changed_to" -> the COMPLETE name only, keeping
  every part of it (ಬಿನ್ / ಕೋಂ / ಲೇಟ್ / ಉರುಫ್ and the following name).
- "liabilities" -> the ಋಣಗಳು text printed in that party's block, VERBATIM. A
  sentence describing a CHARGE or a TRANSFER RESTRICTION (a ಪರಭಾರೆ / ನಿಷೇಧ note,
  typically naming a number of years) also belongs here — never in "relation".
- "relation"    -> ONLY the short value printed immediately after ಸಂಬಂಧ : in that
  party's block (a single word or two naming the relationship). If nothing
  follows ಸಂಬಂಧ :, or the label is absent, "relation" MUST be null. A full
  sentence is NEVER a relation.
- A ಋಣಗಳು note or a ಸಂಬಂಧ annotation is NEVER a name and NEVER gets an entry of
  its own. Giving one its own row also pushes the ವಿಸ್ತೀರ್ಣ values onto the wrong
  parties, which corrupts every later extent — so never do it.
- Each ವಿಸ್ತೀರ್ಣ belongs to the PARTY NAME printed beside it, not to a ಋಣಗಳು or
  ಸಂಬಂಧ line.

JOINT holders (very important). One side may list SEVERAL people who jointly
hold ONE transaction over a SINGLE printed extent. The extra names are marked
"ಮೇಲಿನ ಜಂಟಿ" ("joint with the one above") and have NO extent of their own.
Handle this as follows:
- "ಮೇಲಿನ ಜಂಟಿ" is a JOINT MARKER, not data. NEVER put it in a name field and
  NEVER put it in an extent field.
- "ಸಂಬಂಧ : <x>" is a relationship annotation, not part of the name. Keep it out of
  rights_changed_from / rights_changed_to and record it in "relation" instead.
- Emit ONE entry per transaction, not one per joint name. Join the joint people
  into that side's single name field separated by " / " (e.g.
  "A ಬಿನ್ X / B ಬಿನ್ Y"), and put the shared extent ONCE in the matching extent
  field. Do NOT create extra entries whose only content is a joint name or the
  word ಮೇಲಿನ ಜಂಟಿ.
- ONLY join names this way when the ಮೇಲಿನ ಜಂಟಿ marker is ACTUALLY PRINTED for the
  extra names. If each name has its OWN printed extent, the names are SEPARATE
  parties on SEPARATE rows — emit a separate entry for each and NEVER join them
  with " / ". When in doubt, keep them separate.

Some Mutation Register copies instead use a two-part table. Capture BOTH parts.

Part 1 — "ಈಗಿನ ಸರ್ವೆ ನಂಬರಿನ ವಿವರ" (current survey summary). Put this in the
top-level "survey_details" object (NOT in "entries"). Read from the IMAGE:
- "survey_number" -> ಸರ್ವೆ ನಂಬರು
- "total_extent"  -> ಒಟ್ಟು under ಖೇತುವಾರು ವಿಸ್ತೀರ್ಣ ವಿವರ
- "kharab_a"      -> ಖರಾಬ್(ಅ), when printed
- "kharab_b"      -> ಖರಾಬ್(ಬ)
- "revenue"       -> ಕಂದಾಯ (under ಭೂಕಂದಾಯ ವಿವರ)
- "judi"          -> ಜೂಡಿ
- "cess"          -> ಸೆಸ್ಸು
- "water_rate"    -> ನೀರಿನ ದರ
- "patta"         -> ಪಟ್ಟಾ
- "soil_type"     -> ಮಣ್ಣಿನ ನಮೂನೆ
Leave "survey_details" all-null for the common transferor/transferee layout,
which has no such summary block.

Part 2 — "ಸರ್ವೆ ನಂಬರಿನಲ್ಲಿ ಸ್ವಾಧೀನದಾರರು ಮತ್ತು ಅವರ ವಿಸ್ತೀರ್ಣ" (possessors in the
survey). Create one entry per possessor ROW and map:
- "survey_hissa_number" -> ಸರ್ವೆ ನಂಬರು
- "rights_changed_to"   -> ಸ್ವಾಧೀನದಾರರ ಹೆಸರು (the possessor/holder name)
- "extent_to"           -> ವಿಸ್ತೀರ್ಣ
- "khata_number"        -> ಖಾತೆ ನಂಬರು
- "possession_mode"     -> ಸ್ವಾಧೀನತೆಯ ರೀತಿ (the acquisition mode / date / MR
                           reference for that row)
- "rights"              -> ಹಕ್ಕುಗಳು (the rights text for that row)
- "liabilities"         -> ಋಣಗಳು (the debts / charges / encumbrances column —
                           any loan or charge recorded against the holding).
                           Capture it VERBATIM; do NOT merge it into
                           possession_mode or drop it. Leave null only when the
                           ಋಣಗಳು cell is truly empty.
- "row_status"          -> ಸ್ಥಿತಿ (the row's status). This column exists ONLY in
                           this two-part layout. In the common transferor/
                           transferee layout there is no ಸ್ಥಿತಿ column, so
                           "row_status" MUST be null there — never fill it from a
                           ಷರಾ note or any other cell.
Leave the fields that do not apply to the layout as null.

Footer:
- "mutation_order_details" -> ಮ್ಯುಟೇಶನ್ ಆದೇಶದ ವಿವರ: the full free-text paragraph
  describing the mutation order. Capture it VERBATIM in Kannada, as one string.
- "place"      -> the place printed under ಮ್ಯುಟೇಶನ್ ಆದೇಶದ ವಿವರ / ಮೊಕ್ಕಾಂ in the
                  footer.
- "order_date" -> ಮ್ಯುಟೇಶನ್ ಆದೇಶದ ದಿನಾಂಕ printed in the footer. Distinct from
                  "order_reference_date".
- "signatory"  -> the signing-officer line at the bottom-right, typically
                  ರಾಜಸ್ವ ನಿರೀಕ್ಷಕರು ಹೋಬಳಿ <hobli> (Revenue Inspector). Capture
                  the role/hobli text; ignore the ಸಹಿ/- ("Sd/-") token.

Rules:
- Preserve Kannada text EXACTLY in Kannada Unicode. Do NOT transliterate and do
  NOT translate.
- Create exactly one entry object per data row in the table, in order. If the
  table has a single mutation row, "entries" has a single object.
- Keep the dotted extent format EXACTLY as printed. Do not reformat it.
- A grey diagonal "View Only" watermark crosses the page. It is translucent —
  read the text/digit printed under it. Do NOT guess a value the watermark
  genuinely hides; leave that field null instead of inventing one.
- Use null for any field that is genuinely absent or illegible. Do NOT fabricate
  a value, and do NOT copy a column heading in as a value.

CRITICAL - name and place fields (rights_changed_from, rights_changed_to,
signatory, district, taluk, hobli, village, and the order paragraph):
- The IMAGE is your PRIMARY and authoritative source. These copies are usually
  crisp, digitally generated PDFs, so the printed Kannada is clean. Read each
  name from the image.
- An OCR TEXT DUMP of this exact document is provided below the image. Treat it
  as a NOISY cross-check, NOT the source of truth. This pipeline's OCR
  frequently drops a leading initial, breaks a conjunct, or splits one name
  across lines.
- Reconcile the image against the OCR per name:
  (a) They agree -> use it.
  (b) The image is clearly legible and the OCR differs only by a dropped
      initial, a broken/garbled conjunct, or a split fragment -> TRUST THE
      IMAGE; do NOT copy the OCR's damage. NEVER drop a leading initial or
      letter that the image shows.
  (c) The image glyphs are genuinely blurry/ambiguous AND the OCR shows a
      clear, plausible Kannada name -> prefer the OCR reading.
  (d) NEVER replace a name with a different, more common-sounding name that
      NEITHER source supports.
- A relation word (ಬಿನ್ / ಕೋಂ / ತಂದೆ / ಉರುಪ್ / ಗುಳಪ್ಪ) inside a party name is
  part of that name — keep it in the same field; do not split it into a
  separate field.
- If a name is printed in Latin/English letters, keep it VERBATIM in Latin
  script. Do NOT transliterate it into Kannada.
"""


OCR_GROUNDING_HEADER = (
    "OCR TEXT DUMP of this exact document. This is a NOISY cross-check only —\n"
    "the IMAGE is authoritative for names, the survey/reference codes, and\n"
    "dates. Use this text to catch wholesale name hallucination, not to override\n"
    "clear image glyphs (see the name-reconciliation rules above):\n"
    "--- BEGIN OCR TEXT ---\n"
)

OCR_GROUNDING_FOOTER = "\n--- END OCR TEXT ---\n"


# Tried in order. The Gemini free tier enforces a per-model daily request quota,
# so when the primary model is exhausted (429 RESOURCE_EXHAUSTED) the next model
# has its own separate quota bucket. This mirrors the PAN / Aadhaar / Patta
# extractors and is the fix for the per-model daily-quota flakiness.
#
# ACCURACY-FIRST ORDER: lead with the strongest reader (2.5-pro) for the best
# Kannada glyph accuracy on the party-name cells, then fall through progressively
# cheaper flash models. Each entry is a SEPARATE daily-quota bucket, so this list
# also widens quota resilience: only when EVERY model here returns nothing does
# the pipeline fall to the OCR-table fallback (which cannot read Kannada names).
# A prior two-model list collapsed to that OCR garbage the moment both 429'd.
# Every model here was verified callable on the project API key (the API's
# model-list also returns models that 404 for this key, e.g. gemini-2.5-pro and
# the gemini-2.0-flash* pair — do NOT add a model without confirming it answers).
# VERIFIED CALLABLE on this project key 2026-08-31. The previous list was
# entirely / partly RETIRED — Google returns 404 "no longer available to new
# users" for every gemini-2.5-* and gemini-2.0-* name, so those entries could
# never answer and each burned 3 retries before the next was tried. Google's own
# migration targets: 2.5-flash and 2.0-flash -> 3.6-flash, 2.5-flash-lite and
# 2.0-flash-lite -> 3.5-flash-lite. Do NOT add a name without confirming it
# answers on this key; models.list() also returns names that 404.
MR_MODELS = [
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


# A Mutation Register's holder table routinely spills across several pages. Send
# EVERY page image to Gemini (not just page 1) so rows on later pages are read
# from the image — the authoritative source — instead of the frequently-garbled
# OCR text. Capped so a pathologically long scan can't blow the request budget;
# the cap is logged, never silent.
_MAX_PDF_PAGES = 15


def _pdf_to_images(pdf_path: str, dpi: int = 240) -> list:
    pdf = fitz.open(pdf_path)
    page_count = len(pdf)

    images = []
    for page_index in range(min(page_count, _MAX_PDF_PAGES)):
        pix = pdf[page_index].get_pixmap(dpi=dpi)
        images.append(
            Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        )

    pdf.close()

    if page_count > _MAX_PDF_PAGES:
        print(
            f"Gemini MR: PDF has {page_count} pages; sending the first "
            f"{_MAX_PDF_PAGES} page images (later pages read from OCR only)."
        )

    return images


def _load_images(file_path: str) -> list:
    if file_path.lower().endswith(".pdf"):
        return _pdf_to_images(file_path)
    return [Image.open(file_path).convert("RGB")]


def _clean_str(value):
    if value is None:
        return None

    value = re.sub(r"\s+", " ", str(value)).strip()

    return value or None


# Per-entry fields, in output order.
_ENTRY_FIELDS = (
    "survey_hissa_number",
    "patta_number",
    "revenue",
    "rights_changed_from",
    "extent_from",
    "rights_changed_to",
    "extent_to",
    "relation",
    "khata_number",
    "possession_mode",
    "rights",
    "liabilities",
    "row_status",
    "inspecting_officer",
)

# Survey-summary fields for the two-part layout's "ಈಗಿನ ಸರ್ವೆ ನಂಬರಿನ ವಿವರ" block.
_SURVEY_DETAIL_FIELDS = (
    "survey_number",
    "total_extent",
    "kharab_a",
    "kharab_b",
    "revenue",
    "judi",
    "cess",
    "water_rate",
    "patta",
    "soil_type",
)


# A ಸಂಬಂಧ value is a word or two ("ಇತರೆ", "ಮಗು"). Anything longer is a sentence
# the model swept in from the cell — typically an alienation-restriction note.
_MAX_RELATION_LENGTH = 24
_DIGIT_RE = re.compile(r"\d")


def _repair_entry(entry: dict) -> dict:
    """
    Deterministic repairs for two mistakes the prompt cannot fully prevent, both
    seen on real documents:

    * "patta_number" holding a party name. ಪಟ್ಟಾ is a NUMBER; a value with no
      digit in it is never one, so drop it rather than let a name masquerade as
      a patta number.
    * "relation" holding a whole restriction sentence when ಸಂಬಂಧ : was blank.
      That text is a charge/restriction, so move it to "liabilities" (when empty)
      instead of discarding it.

    Never invents a value; only removes or relocates a clearly-misplaced one.
    """
    patta = entry.get("patta_number")
    if patta and not _DIGIT_RE.search(patta):
        entry["patta_number"] = None

    relation = entry.get("relation")
    if relation and len(relation) > _MAX_RELATION_LENGTH:
        if not entry.get("liabilities"):
            entry["liabilities"] = relation
        entry["relation"] = None

    return entry


def _normalize_entry(entry: dict) -> dict:
    entry = entry if isinstance(entry, dict) else {}
    cleaned = {field: _clean_str(entry.get(field)) for field in _ENTRY_FIELDS}
    return _repair_entry(cleaned)


def _normalize_survey_details(details) -> dict:
    details = details if isinstance(details, dict) else {}
    return {field: _clean_str(details.get(field)) for field in _SURVEY_DETAIL_FIELDS}


def _clean_str_list(values) -> list:
    """
    Normalize a list-valued field, dropping blanks and duplicates while keeping
    the printed order. Tolerates the model returning a single string instead of
    a list.
    """
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    cleaned = []
    for value in values:
        text = _clean_str(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


_KANNADA_RE = re.compile(r"[ಀ-೿]")


def _survey_format_warnings(entries: list, surveys: list) -> list:
    """
    Survey/hissa codes are printed in ASCII (digits, "*", "/", a Latin P hissa).
    A Kannada letter in one means the model transliterated it (or read it from
    the order paragraph, which spells the survey in Kannada). That is always
    wrong, and two models can make the SAME mistake — so the text-agreement pass
    would not catch it. Report the offending codes deterministically.
    """
    suspect = []
    candidates = list(surveys) + [
        entry.get("survey_hissa_number") for entry in entries
    ]
    for code in candidates:
        if code and _KANNADA_RE.search(code) and code not in suspect:
            suspect.append(code)
    return suspect


def _fold_text(value) -> str:
    """
    Fold a value for COMPARISON only (not for output). Drops spaces, dots and
    case, and normalises survey/reference separators ("|", "\\" -> "/"), so that
    trivial formatting differences between two model runs do not count as a
    disagreement while a real character difference still does.
    """
    if value is None:
        return ""
    folded = str(value).replace("\\", "/").replace("|", "/")
    folded = re.sub(r"[\s.‌]", "", folded)
    return folded.casefold()


# Header fields (no arithmetic check) worth cross-verifying between two reads.
_HEADER_COMPARE_FIELDS = (
    "mr_number",
    "disposal_number",
    "source_number",
    "order_reference_date",
    "order_date",
    "status",
)

# Per-entry fields compared directly (both reads should agree cell-for-cell).
_ENTRY_DIRECT_FIELDS = ("survey_hissa_number", "khata_number")

# Party/extent fields are compared as a mapping-agnostic SET, not field-by-field
# (see _row_disagreements).
_PARTY_FIELDS = ("rights_changed_from", "rights_changed_to")
_ROW_EXTENT_FIELDS = ("extent_from", "extent_to")


def _folded_multiset(entry: dict, fields) -> list:
    """Sorted list of the non-empty folded values across `fields`."""
    values = [_fold_text(entry.get(field)) for field in fields]
    return sorted(value for value in values if value)


def _row_disagreements(primary_entry: dict, secondary_entry: dict) -> list:
    """
    Fields on which two reads of the SAME row disagree.

    survey/khata are compared directly. Party names and extents are compared as a
    SET across the from/to slots, NOT field-by-field: on the two-part possessor
    layout the two models often slot the SAME possessor into different columns
    (one as rights_changed_from, the other as rights_changed_to), which a
    positional comparison would wrongly flag on every row. Comparing the set of
    names (and of extents) treats those as agreeing while still catching a
    genuinely different name or extent.
    """
    diffs = [
        field
        for field in _ENTRY_DIRECT_FIELDS
        if _fold_text(primary_entry.get(field)) != _fold_text(secondary_entry.get(field))
    ]
    if _folded_multiset(primary_entry, _PARTY_FIELDS) != _folded_multiset(
        secondary_entry, _PARTY_FIELDS
    ):
        diffs.append("party_names")
    if _folded_multiset(primary_entry, _ROW_EXTENT_FIELDS) != _folded_multiset(
        secondary_entry, _ROW_EXTENT_FIELDS
    ):
        diffs.append("extents")
    return diffs


def _diff_fields(primary: dict, secondary: dict) -> dict:
    """
    Compare two independent extractions on the fields that have no arithmetic
    check, and report where they disagree. This is the analogue of Patta's
    text-verification pass: it cannot PROVE a value is right, but a disagreement
    between two independent reads is a strong signal a field needs a human look.

    LIMITATION: correlated errors (both models make the same glyph mistake) are
    invisible here, so it catches a subset, not all.

    Pure function (no I/O) so the diff logic is unit-testable.
    """
    header_disagreements = [
        field
        for field in _HEADER_COMPARE_FIELDS
        if _fold_text(primary.get(field)) != _fold_text(secondary.get(field))
    ]

    primary_entries = primary.get("entries") or []
    secondary_entries = secondary.get("entries") or []

    entry_disagreements = {}
    rows_compared = min(len(primary_entries), len(secondary_entries))

    # Mutation Register tables are short (often a single row) and share no stable
    # key across reads, so align positionally.
    for index in range(rows_compared):
        differing = _row_disagreements(
            primary_entries[index], secondary_entries[index]
        )
        if differing:
            entry_disagreements[index] = differing

    return {
        "header_disagreements": header_disagreements,
        "entry_disagreements": entry_disagreements,
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
            print(f"Gemini MR {model} attempt {attempt + 1} failed:", e)
            if _is_quota_error(e) or _is_missing_model_error(e):
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

        if response and getattr(response, "text", None):
            return response.text, last_error

    return None, last_error


def _run_mr_models(client, contents, models):
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
            print(f"Gemini MR extracted using model: {model}")
            return text, model, last_error
    return None, None, last_error


def _extract_once(client, contents, model):
    """
    Run ONE model and parse its output into the normalized MR shape. Returns
    (result_dict_or_None, last_error).
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

    parsed = {
        "namune": _clean_str(data.get("namune")),
        "district": _clean_str(data.get("district")),
        "taluk": _clean_str(data.get("taluk")),
        "hobli": _clean_str(data.get("hobli")),
        "village": _clean_str(data.get("village")),
        "disposal_year": _clean_str(data.get("disposal_year")),
        "disposal_number": _clean_str(data.get("disposal_number")),
        "mr_number": _clean_str(data.get("mr_number")),
        "source": _clean_str(data.get("source")),
        "source_number": _clean_str(data.get("source_number")),
        "change_type": _clean_str(data.get("change_type")),
        "possession_type": _clean_str(data.get("possession_type")),
        "order_reference_date": _clean_str(data.get("order_reference_date")),
        "status": _clean_str(data.get("status")),
        "survey_details": _normalize_survey_details(data.get("survey_details")),
        # Every survey/hissa number on the document. On a row-wise table the model
        # may only fill the per-entry field, so fall back to collecting those —
        # this list must never be emptier than what the rows already show.
        "affected_survey_numbers": (
            _clean_str_list(data.get("affected_survey_numbers"))
            or _clean_str_list(
                [entry.get("survey_hissa_number") for entry in entries]
            )
        ),
        "entries": entries,
        "mutation_order_details": _clean_str(data.get("mutation_order_details")),
        "place": _clean_str(data.get("place")),
        "order_date": _clean_str(data.get("order_date")),
        "signatory": _clean_str(data.get("signatory")),
    }

    result = {"model": model, "parsed": parsed}
    return result, error


# Header fields whose presence signals a genuinely-read Mutation Register (as
# opposed to a blank / failed read).
_CORE_HEADER_FIELDS = (
    "mr_number",
    "disposal_number",
    "taluk",
    "hobli",
    "village",
)


# The header keys a Gemini read emits. The OCR fallback fills the same set (null
# where it could not recover a value) so both paths return one identical shape.
_HEADER_OUTPUT_FIELDS = (
    "namune",
    "district",
    "taluk",
    "hobli",
    "village",
    "disposal_year",
    "disposal_number",
    "mr_number",
    "source",
    "source_number",
    "change_type",
    "possession_type",
    "order_reference_date",
    "status",
)


def _ocr_table(file_path: str) -> dict:
    """Best-effort per-cell OCR of the party table. Never raises."""
    try:
        return extract_mr_table_from_image(file_path) or {}
    except Exception as table_error:
        print("MR table OCR fallback failed:", table_error)
        return {}


def _unavailable_result(error, ocr_text: str = None, file_path: str = None) -> dict:
    """
    Shape returned when NO model produced a read (e.g. an exhausted API quota).

    Falls back to the OCR text for the LABELLED HEADER fields, which it prints
    reliably. The party table is deliberately NOT reconstructed: names and extents
    fuse and mangle in this OCR, and a silently-wrong owner in a title record is
    far worse than an absent one. So "entries" stays empty and is flagged for
    manual review, and the reason for the degraded read is reported explicitly.
    """
    quota = error is not None and _is_quota_error(error)
    reason = "quota_exhausted" if quota else "model_unavailable"

    recovered = {}
    try:
        recovered = extract_mutation_register_from_ocr(ocr_text) or {}
    except Exception as fallback_error:  # never let the fallback mask the outage
        print("MR OCR fallback failed:", fallback_error)

    surveys = recovered.pop("affected_survey_numbers", [])
    header_recovered = bool(recovered)

    table = _ocr_table(file_path) if file_path else {}
    entries = [_normalize_entry(entry) for entry in table.get("entries", [])]
    columns = table.get("columns") or {}

    # The flat OCR text usually reads the survey codes better than a cropped
    # cell does, so only fall back to the table's list when it found none.
    if not surveys:
        surveys = columns.get("affected_survey_numbers", [])

    table_recovered = bool(entries or columns)

    # Emit the SAME key set a Gemini read produces, so callers never have to
    # branch on which path ran. Unknown values are null rather than absent.
    result = {
        "document_type": "MUTATION_REGISTER",
        **{field: recovered.get(field) for field in _HEADER_OUTPUT_FIELDS},
        "survey_details": _normalize_survey_details(recovered.get("survey_details")),
        "affected_survey_numbers": surveys,
        "entries": entries,
        "mutation_order_details": recovered.get("mutation_order_details"),
        "place": recovered.get("place"),
        "order_date": recovered.get("order_date"),
        "signatory": recovered.get("signatory"),
        # No model answered, so no cross-check happened. Present the same
        # verification keys a Gemini read carries, with neutral values, so a
        # caller never has to branch on which path produced the record.
        "extraction_attempts": 0,
        "text_verification": {"status": "unavailable", "verified_by": None},
        "header_disagreements": [],
        "entry_disagreements": {},
        "survey_format_warnings": _survey_format_warnings(entries, surveys),
        "ocr_columns": columns,
        "ocr_table_layout": table.get("layout"),
        "extraction_status": (
            "partial_requires_review"
            if (header_recovered or table_recovered) else "unavailable"
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
                " Fields below were recovered by OCR and are LOW CONFIDENCE:"
                " names may carry wrong glyphs and numbers (extents, survey and"
                " khata numbers) are frequently misread. Verify every value"
                " against the document before use."
                if (header_recovered or table_recovered) else ""
            )
        ),
        "error": str(error) if error is not None else None,
        "field_status": {
            "entries": (
                "ocr_fallback_review" if entries else "requires_manual_review"
            ),
            "entries_text": (
                "ocr_fallback_review" if entries else "requires_manual_review"
            ),
            **{field: "ocr_fallback_review" for field in recovered},
        },
        "extraction_source": (
            "ocr_header_and_table_fallback_v1" if table_recovered
            else "ocr_header_fallback_v1" if header_recovered
            else "gemini_vision_mr_v1:unavailable"
        ),
    }
    return result


def _is_complete(parsed: dict) -> bool:
    """
    A read looks complete when the core header identifiers are present AND at
    least one transaction entry was extracted. Used to decide whether to stop
    early or spend another model's quota trying for a fuller read.
    """
    header_present = sum(1 for f in _CORE_HEADER_FIELDS if parsed.get(f)) >= 3
    return header_present and bool(parsed.get("entries"))


def extract_mutation_register_gemini(file_path: str, ocr_text: str = None):
    """
    Extract Karnataka Mutation Register (ನಮೂನೆ-12 / ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್ ಪ್ರತಿ)
    fields from the document using Gemini vision, grounded by the pipeline's OCR
    text.

    Unlike the Patta Book, a Mutation Register has NO printed arithmetic total to
    reconcile against, so there is no numeric checksum here. Trust comes from
    two levers instead: the same per-model daily-quota fallback as the other
    extractors, and an optional second-model text-agreement pass that flags
    header/entry fields where two independent reads disagree.

    Returns a structured dict, or None if Gemini is unavailable or extraction
    fails, so the caller can degrade gracefully.
    """
    if not settings.GOOGLE_API_KEY:
        return None

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    images = _load_images(file_path)

    # Grounding text is optional. When available it is the single biggest
    # accuracy lever for the Kannada name fields.
    contents = [MR_PROMPT]

    if ocr_text and ocr_text.strip():
        contents.append(
            OCR_GROUNDING_HEADER + ocr_text.strip() + OCR_GROUNDING_FOOTER
        )

    # All page images (usually one). Each is authoritative for the rows printed
    # on it; the holder table can continue across pages.
    contents.extend(images)

    # Try models in order. Accept the first COMPLETE read; keep every successful
    # read so a non-accepted one can double as the text-verification sample (no
    # extra call). Availability fallback is preserved: a model that fails
    # entirely is skipped for free.
    max_attempts = (
        settings.MR_VERIFY_MAX_ATTEMPTS if settings.MR_VERIFY_RETRY else 1
    )

    best = None
    successful_results = []
    last_error = None

    for model in MR_MODELS:
        result, error = _extract_once(client, contents, model)
        if error is not None:
            last_error = error
        if result is None:
            continue  # model unavailable/unparseable — try the next, no cost

        successful_results.append(result)

        # Prefer a complete read; otherwise keep the one with the most entries.
        if best is None or (
            not _is_complete(best["parsed"]) and _is_complete(result["parsed"])
        ) or (
            len(result["parsed"]["entries"]) > len(best["parsed"]["entries"])
        ):
            best = result

        if _is_complete(best["parsed"]):
            break  # good enough — accept it
        if len(successful_results) >= max_attempts:
            break  # spent the retry budget; keep the best read so far

    if best is None:
        if last_error:
            print("Gemini MR extraction giving up after all models:", last_error)
        # A Mutation Register can ONLY be read from the image (its Kannada table
        # is not recoverable from the OCR text), so there is no local fallback.
        # Still, report WHY nothing came back — an exhausted API quota looks
        # identical to a genuine extraction failure otherwise.
        return _unavailable_result(last_error, ocr_text, file_path)

    used_model = best["model"]
    parsed = best["parsed"]
    entries = parsed["entries"]
    extraction_attempts = len(successful_results)

    # Text verification: flag header/entry fields where two independent reads
    # disagree. Reuse a different-model read from the loop when one exists (no
    # extra call); only spend an extra call if none is available. Best-effort —
    # never blocks.
    text_verification = {"status": "disabled"}
    header_disagreements = []
    entry_disagreements = {}

    if settings.MR_VERIFY_TEXT_FIELDS:
        spares = [r for r in successful_results if r["model"] != used_model]
        other = spares[0] if spares else None

        if other is None:
            verify_models = [m for m in MR_MODELS if m != used_model]
            verify_text, verify_model, _ = _run_mr_models(
                client, contents, verify_models
            )
            if verify_text:
                verify_data = _extract_json(verify_text)
                if isinstance(verify_data, dict):
                    verify_entries = [
                        _normalize_entry(entry)
                        for entry in (verify_data.get("entries") or [])
                        if isinstance(entry, dict)
                    ]
                    other = {
                        "model": verify_model,
                        "parsed": {**verify_data, "entries": verify_entries},
                    }

        if other is not None:
            diff = _diff_fields(parsed, other["parsed"])
            primary_rows = diff["row_count_primary"]
            secondary_rows = diff["row_count_secondary"]

            # A verifier that under-extracted (returned far fewer rows than the
            # accepted read) is not a trustworthy second opinion: its sparse or
            # degenerate output makes EVERY populated header/entry field look like
            # a disagreement (e.g. a verifier returning 0 rows on a 9-row doc).
            # Report such a pass as "inconclusive" and flag nothing, mirroring the
            # Patta text-verification coverage guard.
            comparable = primary_rows == 0 or secondary_rows >= 0.6 * primary_rows

            if not comparable:
                text_verification = {
                    "status": "inconclusive",
                    "verified_by": other["model"],
                    "row_count": {
                        "primary": primary_rows,
                        "secondary": secondary_rows,
                    },
                    "rows_compared": diff["rows_compared"],
                }
            else:
                header_disagreements = diff["header_disagreements"]
                entry_disagreements = diff["entry_disagreements"]
                has_disagreement = bool(header_disagreements or entry_disagreements)
                text_verification = {
                    "status": "mismatch" if has_disagreement else "ok",
                    "verified_by": other["model"],
                    "header_disagreements": header_disagreements,
                    "entry_disagreements": entry_disagreements,
                    "row_count": {
                        "primary": primary_rows,
                        "secondary": secondary_rows,
                    },
                    "rows_compared": diff["rows_compared"],
                }
        else:
            text_verification = {"status": "unavailable", "verified_by": None}

    header_present = _is_complete(parsed) or sum(
        1 for f in _CORE_HEADER_FIELDS if parsed.get(f)
    ) >= 3

    # A Kannada letter in a survey code is always a misread, whether or not the
    # two models happened to agree on it.
    survey_warnings = _survey_format_warnings(
        entries, parsed.get("affected_survey_numbers") or []
    )

    # Trust the values unless the verification pass ran AND found a disagreement.
    # "disabled"/"unavailable" verification does not by itself force review.
    fields_trustworthy = (
        not (header_disagreements or entry_disagreements) and not survey_warnings
    )

    field_status = {
        field: "extracted" if parsed.get(field) else "requires_manual_review"
        for field in _CORE_HEADER_FIELDS
    }
    field_status["entries"] = (
        "extracted" if entries else "requires_manual_review"
    )
    field_status["entries_text"] = (
        "extracted" if entries and fields_trustworthy
        else "requires_manual_review"
    )
    field_status["survey_numbers"] = (
        "requires_manual_review" if survey_warnings else "extracted"
    )

    return {
        "document_type": "MUTATION_REGISTER",
        **parsed,
        "extraction_attempts": extraction_attempts,
        "text_verification": text_verification,
        "header_disagreements": header_disagreements,
        "entry_disagreements": entry_disagreements,
        "survey_format_warnings": survey_warnings,
        "field_status": field_status,
        "extraction_status": (
            "extracted"
            if header_present and entries and fields_trustworthy
            else "partial_requires_review"
        ),
        "extraction_source": f"gemini_vision_mr_v1:{used_model}",
    }

"""
OCR-text fallback for Mutation Register documents.

Used ONLY when Gemini vision is unavailable (e.g. an exhausted API quota). A
Mutation Register's party table (names + extents) is NOT recoverable from this
pipeline's OCR — the Kannada glyphs fuse, extents lose their dots
("59 25 00 00"), and survey codes lose their slashes ("59//"). Inventing owners
from that would put silently-wrong names into a title record, which is far worse
than omitting them. So this module recovers only what the OCR prints reliably:

  * the labelled header fields (district / taluk / hobli / village, MR number)
  * the disposal year, the dd/mm/yyyy dates, the approval stamp
  * the survey/hissa numbers, which have a distinctive shape

"entries" is always left EMPTY and flagged for manual review. Every value here is
lower-confidence than a Gemini read and the caller marks it as such.
"""

import re


# The label may be followed by ":", "-", "~", "." or nothing, and the value may
# sit on the SAME line (Tesseract) or the NEXT line (EasyOCR).
# field -> (label variants, labels that END this field's value, single-token?)
#
# A single global stop-list does NOT work: possession_type's own value contains
# the word ಬದಲಾವಣೆ ("ಪಹಣಿ ಬದಲಾವಣೆ"), which is another field's label. So each
# field declares only the labels that genuinely follow it on the line.
_LABEL_SPECS = {
    # Multi-token: a Latin district prints as two words ("... Urban").
    "district": (("ಜಿಲ್ಲೆ",), ("ವರ್ಷ", "ವಹಿವಾಟು", "ಮೂಲ", "ಸಂಖ", "ಸ೦ಖ"), False),
    "taluk": (
        ("ತಾಲ್ಲೂಕು", "ತಾಲ್ಕೂಕು"),
        ("ಬದಲಾವಣ", "ವಹಿವಾಟು", "ಪಹಿವಾಟು", "ದಿನಾಂಕ", "ದಿನಾ೦ಕ", "ಸಂಖ", "ಸ೦ಖ"),
        False,
    ),
    "hobli": (
        ("ಹೋಬಳಿ", "ಹೋಬಳ"),
        ("ನಂಬರ್", "ಸ್ವಾಧೀನ", "ಸಾಧೀನ", "ಸ್ಮಾಧೀನ"),
        True,
    ),
    "village": (("ಗ್ರಾಮ", "ಗರಾಮ"), (), True),
    "source": (("ಮೂಲ",), ("ಸಂಖ", "ಸ೦ಖ", "ನೋಂದಣಿ"), False),
    "change_type": (
        ("ಬದಲಾವಣೆ ರೀತಿ", "ಬದಲಾವಣ ರೀತಿ", "ಬದಲಾಪಣೆ ರೀತಿ", "ಬದಲಾವಣ ಈೀತಿ"),
        ("ದಿನಾಂಕ", "ದಿನಾ೦ಕ", "ನೋಂದಣಿ"),
        False,
    ),
    # Its value ends the line, so it stops at nothing.
    "possession_type": (
        ("ಸ್ವಾಧೀನತೆ ರೀತಿ", "ಸಾಧೀನತ ರೀತಿ", "ಸ್ಮಾಧೀನತೆ ರೀತಿ", "ಸಾಧೀನತ ಈೀತಿ"),
        (),
        False,
    ),
    "place": (("ಮೊಕ್ಕಾಂ", "ಮೂಕಾಂ", "ಮೂಕ್ಾಂ"), ("ಸಹಿ",), False),
}

_NAMUNE_RE = re.compile(r"ನಮೂನೆ[\s.\-]*(\d{1,2})")
_SIGNATORY_HINT = "ನಿರೀ"
# OCR mangles ವಿವರ into ಎವರ / ಐವರ / ವವರ, so anchor on "ಆದೇಶದ" plus any of them.
# The optional middle char covers the ಿ vowel sign in the correct spelling.
_ORDER_DETAIL_START_RE = re.compile(r"ಆದೇಶದ\s*[ವಎಐ].{0,1}ವರ")
_ORDER_DETAIL_END = ("ಮೊಕ್ಕಾಂ", "ಮೂಕಾಂ", "ಮೂಕ್ಾಂ")

# Anything that looks like another field label should not be taken as a value.
_ANY_LABEL = re.compile(
    "|".join(
        [
            "ಜಿಲ್ಲೆ", "ತಾಲ್ಲೂಕು", "ತಾಲ್ಕೂಕು", "ಹೋಬಳಿ", "ಹೋಬಳ", "ಗ್ರಾಮ", "ಗರಾಮ",
            "ವಹಿವಾಟು", "ವರ್ಷ", "ಮೂಲ", "ಸಂಖ", "ದಿನಾಂಕ", "ದಿನಾ೦ಕ", "ನಂಬರ್",
            "ಬದಲಾವಣ", "ಸ್ವಾಧೀನ", "ಸಾಧೀನ",
        ]
    )
)

_SEPARATORS = " \t:-~.`>_,;|"

# A survey/hissa code: must carry an asterisk or a P-hissa, which keeps it from
# matching a date like 18/02/2014.
_SURVEY_RE = re.compile(
    r"\b\d{1,4}/(?:\*|[Pp಩ಪ]\w{0,3})(?:/(?:\*|\w{1,4}))?"
)

_DATE_RE = re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b")
_YEAR_RANGE_RE = re.compile(r"\b(\d{4})-(\d{4})\b")
# EasyOCR puts the value on the NEXT line ("MRನಂಬರ್\nH3"), so allow whitespace
# (including a newline) between the label and the number.
_MR_NUMBER_RE = re.compile(r"ನಂಬರ್\W{0,6}([A-Za-z]?[0-9]{1,4})")
# A reference code. Two shapes occur: a lettered code, and a purely numeric
# registration code ("NNNN/NN-NN"). "[" is a common OCR misread of "1" inside one.
_SOURCE_NUMBER_RE = re.compile(
    r"\b[A-Z]{2,}[A-Z0-9()./\-\[\]]*[0-9][A-Z0-9()./\-\[\]]*"
    r"|\b[0-9]{3,5}/[0-9]{2}-[0-9]{2}\b"
)
# ವಹಿವಾಟು ಸಂಖ್ಯೆ — a short running number printed tight against its label. It must
# be anchored to the ಸಂಖ್ಯೆ label (Tesseract writes ಸ೦ಖ with ೦), otherwise the
# nearby ವಹಿವಾಟು ವರ್ಷ year is picked up instead. Capped at 3 digits for the same
# reason.
# EasyOCR wraps the value onto the next line ("ವಹಿವಾಟು\nಸಂಖ್ 10"), so the span
# between label and value may cross a newline. Capped at 3 digits so the adjacent
# ವಹಿವಾಟು ವರ್ಷ year (4 digits) can never be captured instead.
_DISPOSAL_NUMBER_RE = re.compile(
    r"[ವಪ]ಹಿವಾಟು[\s\S]{0,30}?ಸ[ಂ೦]ಖ[^0-9\n]{0,8}([0-9]{1,3})(?![0-9])"
)

_APPROVED = "ಅನುಮೋದಿಸಿದೆ"
_REJECTED = "ತಿರಸ್ಕರಿಸಿದೆ"


# Tesseract renders the anusvara ಂ (U+0C82) as the Kannada digit ೦ (U+0CE6) —
# "ಬೆ೦ಗಳೂರು", "ದಿನಾ೦ಕ". A digit can never appear inside a Kannada word, so a ೦
# directly after a Kannada letter is always a misread anusvara.
_ANUSVARA_FIX_RE = re.compile(r"(?<=[ಀ-೥])೦")


def _clean(value):
    if value is None:
        return None
    value = str(value).translate({0x200B: None, 0x200C: None, 0x200D: None})
    value = _ANUSVARA_FIX_RE.sub("ಂ", value)
    value = re.sub(r"\s+", " ", value).strip(_SEPARATORS + " ")
    return value or None


def _cut_at_stops(tail: str, stops) -> str:
    """Trim the tail at the earliest label that follows this field on the line."""
    end = len(tail)
    for stop in stops:
        position = tail.find(stop)
        if position != -1:
            end = min(end, position)
    return tail[:end]


def _label_value(lines: list, label: str, stops, single_token: bool):
    """
    Value printed after `label`. Tesseract crowds several fields onto one line
    ("ಹೋಬಳಿ - ಜಾಲ2 M.R.ನಂಬರ್ : 33"), so cut the tail at this field's own stop
    labels rather than discarding it. EasyOCR instead puts the value on the
    following line. `single_token` fields are one word.
    """
    for index, line in enumerate(lines):
        position = line.find(label)
        if position == -1:
            continue

        tail = _clean(_cut_at_stops(line[position + len(label):], stops))

        if not tail:
            # Value wrapped onto a following line (EasyOCR splits label/value).
            for next_line in lines[index + 1: index + 3]:
                candidate = _clean(next_line)
                if candidate and not _ANY_LABEL.search(candidate):
                    tail = candidate
                    break

        if not tail:
            continue

        if single_token:
            tail = tail.split()[0]
        return _clean(tail)
    return None


def _first_of(lines, variants, stops, single_token):
    for variant in variants:
        value = _label_value(lines, variant, stops, single_token)
        if value:
            return value
    return None


# The pipeline concatenates each engine's full output. They fail differently
# (Tesseract reads ಜಿಗಣಿ2 / ಬನ್ನೇರುಘಟ್ಟ correctly, EasyOCR reads H3 and ಕ- initials
# correctly), so read BOTH and score the candidates instead of taking whichever
# engine happens to appear first.
_TESSERACT_MARKER = "--- TESSERACT OCR ---"
_EASYOCR_MARKER = "--- EASYOCR OCR ---"

# Fields whose value is Kannada, so a Latin letter in them signals a misread.
_KANNADA_FIELDS = {
    "taluk", "hobli", "village", "source", "change_type",
    "possession_type", "place", "signatory",
}
_JUNK_CHARS = set("()[]{}\"'!$%^&*|~`<>;_+=\\")

# Place names carry no digits (unlike hobli, which is legitimately "ಜಾಲ2").
_DIGIT_FREE_FIELDS = {"district", "taluk", "village", "source", "possession_type"}


def _engine_sections(ocr_text: str):
    """Each engine's own output, Tesseract first. Falls back to the whole text."""
    tesseract_at = ocr_text.find(_TESSERACT_MARKER)
    easyocr_at = ocr_text.find(_EASYOCR_MARKER)

    sections = []
    if tesseract_at != -1:
        end = easyocr_at if easyocr_at > tesseract_at else len(ocr_text)
        sections.append(ocr_text[tesseract_at + len(_TESSERACT_MARKER): end])
    if easyocr_at != -1:
        sections.append(ocr_text[easyocr_at + len(_EASYOCR_MARKER):])

    return sections or [ocr_text]


def _score_candidate(field: str, value: str) -> float:
    """Higher is better. Punishes OCR debris; mildly rewards a fuller reading."""
    if not value:
        return -100.0

    score = min(len(value), 30) * 0.1
    score -= sum(3.0 for ch in value if ch in _JUNK_CHARS)

    if field in _KANNADA_FIELDS:
        score -= sum(1.5 for ch in value if "a" <= ch.lower() <= "z")

    if field in _DIGIT_FREE_FIELDS:
        score -= sum(2.0 for ch in value if ch.isdigit())

    return score


def _best_candidate(field: str, candidates):
    """Pick the highest-scoring reading; ties resolve to the earlier engine."""
    ranked = [c for c in candidates if c]
    if not ranked:
        return None
    return max(ranked, key=lambda value: _score_candidate(field, value))


def _disposal_year(ocr_text):
    r"""
    Pick the TRANSACTION year ("ವಹಿವಾಟು ವರ್ಷ") from the header.

    Taking the first \d{4}-\d{4} in the document reads the wrong year on any
    MR that cites earlier mutations: the liabilities column is dense with
    references like "MR 41/2009-2010" and "MR 145/2004-2005", and they are
    printed ABOVE the header in the OCR text whenever the engine that read the
    header cleanly is not the first one emitted. On H62/2018-19 that produced
    disposal_year "2009-2010" for a 2018-2019 mutation — the header's own
    "2018-2019" was fine in the Tesseract pass, but EasyOCR ran first and read
    it "2018-20]9", which the pattern cannot match.

    A cited year is part of an MR reference and always follows the reference
    number, so it carries a "/" or "-" immediately before it; the header year
    does not. That separator alone removed 32 of this document's 38 matches,
    all of them citations. Among what is left the header year wins on count,
    each OCR pass having read it independently.
    """
    counts = {}
    order = {}

    for match in _YEAR_RANGE_RE.finditer(ocr_text or ""):
        # A citation ("41/2009-2010"); not the header's transaction year.
        if ocr_text[max(0, match.start() - 1):match.start()] in "/-":
            continue

        value = match.group(0)
        counts[value] = counts.get(value, 0) + 1
        order.setdefault(value, match.start())

    if not counts:
        return None

    return max(counts, key=lambda v: (counts[v], -order[v]))


def _signatory(lines):
    """
    The signing-officer line. Prefer one naming the officer (ರಾಜಸ್ವ ನಿರೀಕ್ಷಕರು)
    and reject the order-date line, which also contains ನಿರೀಕ.
    """
    best = None
    for line in lines:
        if _SIGNATORY_HINT not in line:
            continue
        if _ORDER_DETAIL_START_RE.search(line) or any(d in line for d in _DATE_LABELS):
            continue
        cleaned = _clean(line)
        if cleaned and (best is None or len(cleaned) > len(best)):
            best = cleaned
    return best


def _order_details(lines):
    """The ಮ್ಯುಟೇಶನ್ ಆದೇಶದ ವಿವರ paragraph, up to the ಮೊಕ್ಕಾಂ footer."""
    collected = []
    capturing = False
    for line in lines:
        if not capturing:
            match = _ORDER_DETAIL_START_RE.search(line)
            if match:
                capturing = True
                after = _clean(line[match.end():])
                if after:
                    collected.append(after)
            continue
        if any(end in line for end in _ORDER_DETAIL_END):
            break
        collected.append(line.strip())

    text = _clean(" ".join(collected))
    return text if text and len(text) > 15 else None

# On sub-registrar documents the reference date is labelled ನೋಂದಣಿ ದಿನಾಂಕ and OCR
# often drops the ದಿನಾಂಕ half, so anchor on the registration label as well.
_DATE_LABELS = ("ದಿನಾಂಕ", "ದಿನಾ೦ಕ", "ನೋಂದಣಿ")
# OCR renders the mutation word many ways; match its stable stem.
_MUTATION_STEM_RE = re.compile(r"ಮ.{0,3}ಟ.{0,2}ಶನ್")


def _anchored_dates(lines: list) -> dict:
    """
    Dates anchored to their ದಿನಾಂಕ label, NOT guessed by position.

    A Mutation Register prints two: the order's reference date in the header, and
    the mutation-order date in the footer. Position alone is unreliable — the OCR
    text concatenates several engines, and the free-text order paragraph carries
    unrelated historical dates. So: a ದಿನಾಂಕ line whose own or preceding lines
    mention the mutation word is the FOOTER date; otherwise it is the header
    reference date.
    """
    # MUST be called per ENGINE SECTION. Across the concatenated text the footer
    # of one engine is followed by the header of the next, which would make the
    # header's reference date look like a footer order date.
    dates = []
    for line in lines:
        if not any(label in line for label in _DATE_LABELS):
            continue
        match = _DATE_RE.search(line)
        if match:
            dates.append("/".join(match.groups()))

    if not dates:
        return {}

    # Within one engine's output the header reference date prints first and the
    # footer ಮ್ಯುಟೇಶನ್ ಆದೇಶದ ದಿನಾಂಕ prints last. Historical dates quoted inside the
    # order paragraph are written "ದಿ.<date>", not on a ದಿನಾಂಕ-labelled line, so
    # they do not appear here.
    found = {"order_reference_date": dates[0]}
    if dates[-1] != dates[0]:
        found["order_date"] = dates[-1]
    return found


# Bracket glyphs OCR routinely substitutes for the digit 1.
_ONE_GLYPHS = str.maketrans({"[": "1", "]": "1", "l": "1", "I": "1", "|": "1"})
# Separators OCR emits for "/" — including "." in the "ದಿ.20/10/2005" citation form.
_LOOSE_DATE_RE = re.compile(r"([0-9\[\]lI|]{2})\s*[/\-.]\s*([0-9\[\]lI|]{2})\s*[/\-.]\s*([0-9]{4})")
# The same date with its first separator read as a digit 1: 10109/2008 -> 10/09/2008.
_FUSED_DATE_RE = re.compile(r"\b([0-9]{2})1([0-9]{2})[/\-]([0-9]{4})\b")


def _footer_order_date(lines: list) -> str:
    """
    The signing date below the mutation-order paragraph, found without relying on
    its ದಿನಾಂಕ label.

    _anchored_dates needs that label, but OCR drops or mangles it (ಏನಂತ) and can
    read the date's own separator as a digit. Both failures are confined to the
    footer, so scan only from the order-details marker down and take the LAST
    date: the paragraph quotes historical dates before the signature line.
    """
    capturing = False
    last = None
    for line in lines:
        if not capturing:
            capturing = bool(_ORDER_DETAIL_START_RE.search(line))
            continue
        for match in _LOOSE_DATE_RE.finditer(line):
            day, month, year = (part.translate(_ONE_GLYPHS) for part in match.groups())
            last = f"{day}/{month}/{year}"
        for match in _FUSED_DATE_RE.finditer(line):
            last = "/".join(match.groups())
    return last


def extract_mutation_register_from_ocr(ocr_text: str) -> dict:
    """
    Best-effort header extraction from the OCR text. Returns a dict of the
    fields it could recover (missing ones are simply absent), plus
    "affected_survey_numbers". Never returns party entries.
    """
    if not ocr_text or not ocr_text.strip():
        return {}

    sections = _engine_sections(ocr_text)
    section_lines = [
        [line for line in section.splitlines() if line.strip()]
        for section in sections
    ]
    lines = [line for line in ocr_text.splitlines() if line.strip()]

    recovered = {}

    # Read every engine, then keep the best-scoring reading of each field.
    for field, (variants, stops, single_token) in _LABEL_SPECS.items():
        candidates = [
            _first_of(engine_lines, variants, stops, single_token)
            for engine_lines in section_lines
        ]
        value = _best_candidate(field, candidates)
        if value:
            recovered[field] = value

    for field, engine_values in (
        ("signatory", [_signatory(el) for el in section_lines]),
        ("mutation_order_details", [_order_details(el) for el in section_lines]),
    ):
        value = _best_candidate(field, engine_values)
        if value:
            recovered[field] = value

    namune = _NAMUNE_RE.search(ocr_text)
    if namune:
        recovered["namune"] = namune.group(1)

    year = _disposal_year(ocr_text)
    if year:
        recovered["disposal_year"] = year

    # An MR number may carry a letter prefix (e.g. an "H" or "T"). Only one engine
    # tends to keep it, and a bare number is what you get when the prefix is lost,
    # so prefer a reading that HAS a letter over one that does not.
    mr_candidates = [
        match.group(1)
        for match in (_MR_NUMBER_RE.search(section) for section in sections)
        if match
    ]
    if mr_candidates:
        recovered["mr_number"] = max(
            mr_candidates, key=lambda code: (any(c.isalpha() for c in code), len(code))
        )

    disposal = [
        match.group(1)
        for match in (_DISPOSAL_NUMBER_RE.search(section) for section in sections)
        if match
    ]
    if disposal:
        recovered["disposal_number"] = disposal[0]

    # Dates per engine, then keep the first engine that resolved each one.
    for engine_lines in section_lines:
        for field, value in _anchored_dates(engine_lines).items():
            recovered.setdefault(field, value)

    # Only where no engine's ದಿನಾಂಕ label survived: recover the order date from the
    # footer block by position instead.
    if not recovered.get("order_date"):
        for engine_lines in section_lines:
            footer = _footer_order_date(engine_lines)
            if footer and footer != recovered.get("order_reference_date"):
                recovered["order_date"] = footer
                break

    if _APPROVED in ocr_text:
        recovered["status"] = _APPROVED
    elif _REJECTED in ocr_text:
        recovered["status"] = _REJECTED

    # The longest uppercase alphanumeric code is the ಸಂಖ್ಯೆ reference number.
    # "[" and "]" are OCR artefacts of "1" inside such a code — they can never
    # legitimately appear, so normalise them the way "|" -> "/" is handled.
    codes = []
    for section in sections:
        for code in _SOURCE_NUMBER_RE.findall(section):
            if any(ch.isdigit() for ch in code):
                codes.append(code.replace("[", "1").replace("]", "1"))
    if codes:
        recovered["source_number"] = max(codes, key=len)

    surveys = []
    for match in _SURVEY_RE.findall(ocr_text):
        if match not in surveys:
            surveys.append(match)
    recovered["affected_survey_numbers"] = surveys

    return recovered

import re
from typing import Dict, Any, Optional

from app.services.kyc.aadhaar_qr import read_aadhaar_qr

AADHAAR_REGEX = r"\b\d{4}[\s_]*\d{4}[\s_]*\d{4}\b"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


# Below this many occurrences across the whole OCR text, the best candidate is
# indistinguishable from noise.
#
# 3 was set just above the "every candidate scored 2" floor seen on a bare
# page + embedded-image read. The enhanced pass has since grown to seven
# preprocessing variants (gray/clahe/otsu/adaptive/sharp, psm 11, psm 12), and
# a SYSTEMATIC misread repeats across them exactly as a real name does: one
# engine failing the same way on the same image is not corroboration, but
# counting cannot tell the two apart. On C NAGARAJU — a card whose embedded
# images are genuinely too small to read — Tesseract produced "Ge veer" in the
# embedded-image pass and again under psm 11 and psm 12, reaching 3 and being
# reported as the holder's name, while the true name ("Negara:", a misread of
# C Nagaraju) scored 2. EasyOCR never saw "Ge veer" at all.
#
# Genuine names clear this by a wide margin — 20 (GIRIJAMMA) and 10
# (RAMALINGAPPA) in the recorded samples — so raising the floor above what one
# engine's pass set can manufacture costs the readable cards nothing.
_MIN_NAME_SUPPORT = 5


def extract_crop_field(text: str, field_name: str) -> Optional[str]:
    # Leading \s* is REQUIRED, not cosmetic. documents.py builds this block from
    # an indented f-string, so every line arrives as "        NAME: value".
    # Anchoring "^" straight at the field name matched nothing, which silently
    # disabled the ENTIRE crop-OCR path — every crop field came back None and
    # all values fell through to full-page OCR, while the result still reported
    # extraction_source "aadhaar_parser_v4_crop_primary".
    match = re.search(
        rf"^\s*{field_name}\s*:\s*(.*)$",
        text or "",
        flags=re.IGNORECASE | re.MULTILINE
    )

    if not match:
        return None

    value = clean_text(match.group(1))
    return value if value else None


def extract_aadhaar_number(text: str) -> Optional[str]:
    if not text:
        return None

    match = re.search(AADHAAR_REGEX, text)

    if not match:
        return None

    number = re.sub(r"\D", "", match.group())

    if len(number) != 12:
        return None

    return f"{number[:4]} {number[4:8]} {number[8:]}"


def extract_dob_or_yob(text: str) -> Optional[str]:
    if not text:
        return None

    # Validate the RANGES, do not just take the first thing shaped like a date.
    # OCR misreads single digits, and an unvalidated match reports impossible
    # dates as fact: a low-resolution card gave "31/00/1979" (month 00) and it
    # was returned with field_status "extracted", so nothing downstream could
    # tell it was wrong. Scan every candidate and keep the first VALID one; if
    # none is valid the date is unknown, which is the honest answer.
    for match in re.finditer(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b", text):
        day, month, year = (int(part) for part in match.groups())

        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
            return match.group().replace("-", "/")

    yob = re.search(
        r"(?:YEAR OF BIRTH|YOB)\s*[:\-]?\s*(\d{4})",
        text,
        re.IGNORECASE
    )

    if yob:
        return yob.group(1)

    return None


def extract_gender(text: str) -> Optional[str]:
    upper = (text or "").upper()

    if "FEMALE" in upper:
        return "FEMALE"

    if "MALE" in upper:
        return "MALE"

    if "TRANSGENDER" in upper:
        return "TRANSGENDER"

    return None


def normalize_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = re.sub(r"[^A-Za-z ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip().upper()

    if not value:
        return None

    bad_words = {
        "PAGE", "OCR", "TESSERACT", "EASYOCR", "SELECTED", "ENGINE",
        "GOVERNMENT", "INDIA", "AADHAAR", "UNIQUE", "IDENTIFICATION",
        "AUTHORITY", "ADDRESS", "DOB", "YOB", "YEAR", "BIRTH",
        "MALE", "FEMALE", "TRANSGENDER", "SIGNATURE", "UIDAI",
        "GENDER", "HUSBAND", "FATHER", "WIFE", "HELP"
    }

    words = [
        word
        for word in value.split()
        if word not in bad_words
    ]

    if not words:
        return None

    return " ".join(words)


def valid_aadhaar_name(value: Optional[str]) -> Optional[str]:
    value = normalize_name(value)

    if not value:
        return None

    words = value.split()

    # allow single-word names like GIRIJAMMA
    if len(words) == 1:
        return value if len(words[0]) >= 4 else None

    if len(words) > 4:
        return None

    # reject OCR garbage like PC FO COMO RAJU
    two_letter_words = [w for w in words if len(w) == 2]

    if len(two_letter_words) >= 2:
        return None

    if not any(len(w) >= 4 for w in words):
        return None

    # Printed card wording and pipeline markers — labels, never a holder's name.
    bad_words = {
        "COMO", "FELFALE", "DOA", "PAGE", "OCR",
        "TESSERACT", "EASYOCR", "SELECTED", "ENGINE",
        "GOVERNMENT", "INDIA", "AADHAAR", "UNIQUE",
        "IDENTIFICATION", "AUTHORITY", "ADDRESS", "DETAILS",
        "MALE", "FEMALE", "HUSBAND", "FATHER", "WIFE",
        "ENHANCED", "CROP", "FIELD", "NUMBER", "GENDER",
        "PREPROCESS", "VARIANT", "PROOF", "IDENTITY",
        "CITIZENSHIP", "BIRTH", "VERIFICATION", "SCANNING",
        "OFFLINE", "ONLINE", "AUTHENTICATION",
        "USED", "ONLY", "SHOULD", "SCANNING", "CODE", "XML",
    }

    if any(w in bad_words for w in words):
        return None

    return value


def extract_name_candidates(text: str) -> list:
    lines = [
        clean_text(line)
        for line in (text or "").splitlines()
        if clean_text(line)
    ]

    ignore_phrases = {
        "PAGE OCR",
        "SELECTED OCR ENGINE",
        "TESSERACT OCR",
        "EASYOCR OCR",
        "ENHANCED AADHAAR OCR",
        "AADHAAR FIELD CROP OCR",
    }

    candidates = []

    for line in lines:
        upper = line.upper()

        if any(phrase in upper for phrase in ignore_phrases):
            continue

        name = valid_aadhaar_name(line)

        if name and name not in candidates:
            candidates.append(name)

    return candidates


def extract_name(text: str) -> Optional[str]:
    candidates = extract_name_candidates(text)

    return candidates[0] if candidates else None


def choose_aadhaar_name(
    text: str,
    crop_name: Optional[str],
    guardian_name: Optional[str] = None
) -> Optional[str]:
    """
    Pick the holder name that the OCR text CORROBORATES.

    A single crop read is not trustworthy on its own: the crop pipeline scores
    whole profiles (name+dob+gender+number together), so it can select a
    profile that read the date well and the name badly, discarding another
    profile that had the name exactly right. Both observed —
    "Ramalingappa" was read perfectly by one profile and dropped in favour of
    "Runnings" from a higher-scoring one.

    A genuine name is printed more than once across the card and the several
    OCR passes (in the real samples the correct names appeared 10 and 20 times,
    their garbled rivals once), so occurrence count across the whole text is a
    strong, content-free signal. The crop read still wins ties, preserving its
    priority where nothing contradicts it.
    """
    candidates = []

    if crop_name:
        candidates.append(crop_name)

    for candidate in extract_name_candidates(text):
        if candidate not in candidates:
            candidates.append(candidate)

    # The guardian is printed on a LABELLED line ("Husband :", "W/O", ...), so
    # it is known by name rather than guessed — and whoever that is, they are
    # not the holder. Excluding them is the one content-free fact available
    # here, and it has to be applied before ranking because support alone gets
    # this backwards: the guardian is set in Latin capitals and survives a bad
    # page scan that mangles the holder's mixed-case line, so it can out-count
    # the holder outright rather than merely tying. On GIRIJAMMA's card the
    # husband BASAVARAJU scored 5 to her 3, and the count-then-position rule
    # returned the husband as the holder.
    if guardian_name:
        candidates = [c for c in candidates if c != guardian_name]

    if not candidates:
        return None

    haystack = (text or "").upper()

    def support(candidate: str) -> int:
        return haystack.count(candidate.upper())

    def first_position(candidate: str) -> int:
        index = haystack.find(candidate.upper())
        return index if index >= 0 else len(haystack)

    # Ties are common and must break on LAYOUT, not luck. An Aadhaar prints the
    # holder's name at the top, above the relationship line and above the
    # disclaimer paragraph, so on a tie the earliest occurrence is the holder.
    # Both real ties needed this: GIRIJAMMA tied 20-20 with the husband's name
    # BASAVARAJU, and RAMALINGAPPA tied 10-10 with the disclaimer fragment
    # "USED ONLY".
    best = max(
        candidates,
        key=lambda candidate: (
            support(candidate),
            -first_position(candidate),
            candidate == crop_name,
        )
    )

    # A genuine name is printed on the card and re-read by several OCR passes,
    # so it accumulates support. When even the best candidate barely appears,
    # every candidate is OCR noise and there is no name to report — a card too
    # low-resolution to read must come back None and be flagged for review,
    # never a confident-looking string like "JIXNY OR INULS".
    if support(best) < _MIN_NAME_SUPPORT:
        return None

    return best


def _card_images(file_path: str):
    """Every image worth searching for a QR: each PDF page's embedded images
    and its render, or the image file itself."""
    import numpy as np
    from PIL import Image

    if not file_path:
        return []

    if not file_path.lower().endswith(".pdf"):
        try:
            return [np.array(Image.open(file_path).convert("RGB"))]
        except Exception:
            return []

    images = []

    try:
        import fitz
        import io as _io

        document = fitz.open(file_path)

        for page in document:
            for info in page.get_images(full=True):
                try:
                    raw = document.extract_image(info[0])["image"]
                    images.append(
                        np.array(Image.open(_io.BytesIO(raw)).convert("RGB"))
                    )
                except Exception:
                    continue

            try:
                pix = page.get_pixmap(dpi=400)
                images.append(np.array(
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ))
            except Exception:
                pass

        document.close()
    except Exception as error:
        print("Aadhaar QR image collection failed:", error)

    return images


# Fields the QR carries that map onto extracted fields, for gap-filling and
# for comparison.
_QR_COMPARABLE = (
    "aadhaar_number", "name", "date_of_birth_or_year",
    "gender", "guardian_name", "pin_code", "address",
)


def _fold_for_compare(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def extract_aadhaar_card(
    text: str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    crop_number_raw = extract_crop_field(text, "AADHAAR_NUMBER")
    crop_name_raw = extract_crop_field(text, "NAME")
    crop_dob_raw = extract_crop_field(text, "DOB")
    crop_gender_raw = extract_crop_field(text, "GENDER")

    crop_number = extract_aadhaar_number(crop_number_raw or "")
    crop_name = valid_aadhaar_name(crop_name_raw)
    crop_dob = extract_dob_or_yob(crop_dob_raw or "")
    crop_gender = extract_gender(crop_gender_raw or "")

    full_number = extract_aadhaar_number(text)
    full_name = extract_name(text)
    full_dob = extract_dob_or_yob(text)
    full_gender = extract_gender(text)

    aadhaar_number = crop_number or full_number

    dob = crop_dob or full_dob
    gender = crop_gender or full_gender

    vid = extract_vid(text)
    # Resolved BEFORE the holder name so the guardian can be excluded from the
    # holder candidates; the crop read is not exempt, having itself returned
    # the husband's name on GIRIJAMMA's card.
    guardian = extract_guardian(text)
    address = extract_address(text)

    # Crop name is the most reliable source, so it WINS when present — but it
    # must not be the only source. full_name was computed above and then
    # discarded, so a card whose name crop failed reported name=null even
    # though the full-page text contained it. Falling back does not overwrite a
    # crop read; it only fills the gap, matching how number/dob/gender behave.
    name = choose_aadhaar_name(text, crop_name, guardian["guardian_name"])

    if not guardian["guardian_name"]:
        guardian["guardian_name"] = guardian_from_address(address)
    pin_code = extract_pin_code(address or text)
    issued_on = extract_issue_or_details_date(text)

    fields = {
        "aadhaar_number": aadhaar_number,
        "vid": vid,
        "name": name,
        "guardian_relation": guardian["guardian_relation"],
        "guardian_name": guardian["guardian_name"],
        "date_of_birth_or_year": dob,
        "gender": gender,
        "address": address,
        "pin_code": pin_code,
        "issued_on": issued_on,
    }

    # A field is only "extracted" when it was actually read. The core four are
    # required for a complete card; VID, guardian, address, PIN and the issue
    # date are printed on the back or on some layouts only, so their absence
    # marks the read partial rather than failed.
    core = ("aadhaar_number", "name", "date_of_birth_or_year", "gender")

    field_status = {
        key: "extracted" if value else "requires_manual_review"
        for key, value in fields.items()
    }

    # The QR carries these fields as exact text, with no OCR error possible.
    # It is used to FILL what OCR could not read and to CROSS-CHECK what it
    # could — never to overwrite. A real card here prints DOB 01/01/1952 while
    # its QR encodes 01/01/1972 (verified against the pixels), so overwriting
    # would have silently replaced a correct value with a wrong one.
    qr_data = None
    qr_disagreements = []

    if image_path:
        try:
            qr_data = read_aadhaar_qr(_card_images(image_path))
        except Exception as error:
            print("Aadhaar QR read failed:", error)

    if qr_data:
        for key in _QR_COMPARABLE:
            qr_value = qr_data.get(key)

            if not qr_value:
                continue

            if not fields.get(key):
                fields[key] = qr_value
            elif _fold_for_compare(fields[key]) != _fold_for_compare(qr_value):
                qr_disagreements.append({
                    "field": key,
                    "printed": fields[key],
                    "qr": qr_value,
                })

        # Address components the printed block does not break out.
        for key in ("district", "state", "post_office", "vtc"):
            if qr_data.get(key):
                fields.setdefault(key, qr_data[key])

    return {
        "document_type": "AADHAAR_CARD",
        **fields,
        "qr_verified": bool(qr_data),
        "qr_disagreements": qr_disagreements,
        "field_status": field_status,
        "extraction_status": (
            "extracted"
            if all(fields[key] for key in core)
            else "partial_requires_review"
        ),
        "extraction_source": "aadhaar_parser_v4_crop_primary",
    }

# ---------------------------------------------------------------------------
# Remaining printed fields: VID, guardian (S/O, W/O, D/O, C/O), address and its
# components, and the issue / details date.
#
# Everything below keys on STRUCTURE — digit grouping, UIDAI's printed relation
# markers, the "Address:" label, and position relative to the PIN code. No
# sample values, no village/district/state lists: a fixed place list would snap
# real addresses onto whichever places happened to be in the samples.
# ---------------------------------------------------------------------------

# UIDAI prints the guardian with one of these markers, in English or the local
# language. These are the form's own printed labels, not extracted values.
_RELATION_MARKERS = [
    (r"S\s*/\s*O", "S/O"),
    (r"D\s*/\s*O", "D/O"),
    (r"W\s*/\s*O", "W/O"),
    (r"C\s*/\s*O", "C/O"),
    (r"FATHER", "S/O"),
    (r"HUSBAND", "W/O"),
    (r"MOTHER", "C/O"),
    (r"WIFE", "W/O"),
    (r"ತಂದೆ", "S/O"),
    (r"ಗಂಡ", "W/O"),
    (r"ಪತಿ", "W/O"),
    (r"ತಾಯಿ", "C/O"),
]

_PIN_PATTERN = re.compile(r"\b([1-9]\d{5})\b")

# A VID is 16 digits, printed in four groups of four.
_VID_PATTERN = re.compile(
    r"VID\s*[:\-]?\s*((?:\d{4}[\s-]*){3}\d{4})",
    re.IGNORECASE
)


def extract_vid(text: str) -> Optional[str]:
    match = _VID_PATTERN.search(text or "")

    if not match:
        return None

    digits = re.sub(r"\D", "", match.group(1))

    if len(digits) != 16:
        return None

    return " ".join(digits[i:i + 4] for i in range(0, 16, 4))


def extract_guardian(text: str) -> Dict[str, Optional[str]]:
    """
    Guardian name plus the relation UIDAI printed for it.

    Read from the marker onwards on the same line, stopping at the first comma
    — the address continues after that on the back of the card.
    """
    # The back of an Aadhaar always opens its address with the care-of line, so
    # when a legible marker is found below it is preferred; this is the
    # fallback for when OCR mangled the marker itself (a real card rendered
    # "S/O:" as "52:", losing the guardian entirely even though the name that
    # follows it came through).
    for pattern, relation in _RELATION_MARKERS:
        match = re.search(
            rf"{pattern}\s*[:\-]?\s*([^\n,]+)",
            text or "",
            re.IGNORECASE
        )

        if not match:
            continue

        candidate = valid_aadhaar_name(match.group(1))

        if candidate:
            return {"guardian_relation": relation, "guardian_name": candidate}

    return {"guardian_relation": None, "guardian_name": None}


def guardian_from_address(address: Optional[str]) -> Optional[str]:
    """
    Care-of name taken from the first component of the address.

    UIDAI prints the address starting with the care-of line, so when the
    relation marker itself is unreadable the NAME after it is still recoverable
    from its position. Relation stays None — we know who, not how they relate.
    """
    if not address:
        return None

    head = address.split(",")[0]

    match = re.match(r"^\s*\S{1,4}\s*[:.]\s*(.+)$", head)

    if not match:
        return None

    return valid_aadhaar_name(match.group(1))


def extract_pin_code(text: str) -> Optional[str]:
    """
    Indian PIN: six digits, never starting with 0.

    Taken from the LAST match, because an Aadhaar back prints the address twice
    (local language then English) and the English block — the one whose other
    components are parsed below — comes second.
    """
    matches = _PIN_PATTERN.findall(text or "")

    return matches[-1] if matches else None


def _clean_address_lines(segment: str) -> Optional[str]:
    """Turn a raw text segment into one address line, stopping at the PIN."""
    parts = []

    for line in segment.splitlines():
        line = clean_text(line)

        if not line:
            continue

        latin = len(re.findall(r"[A-Za-z]", line))
        kannada = len(re.findall(r"[ಀ-೿]", line))
        digits = len(re.findall(r"\d", line))

        if latin + digits < 3:
            continue

        # Drop the local-language copy: UIDAI prints the address twice and OCR
        # interleaves the two line by line.
        if latin < kannada:
            continue

        if re.search(r"-{2,}|OCR ENGINE|TESSERACT|EASYOCR|ENHANCED", line, re.I):
            continue

        parts.append(line)

        if _PIN_PATTERN.search(line):
            break

    if not parts:
        return None

    address = ", ".join(parts)
    address = re.sub(r"\s*,\s*", ", ", address)
    address = re.sub(r",\s*,+", ", ", address).strip(" ,")

    pin = _PIN_PATTERN.search(address)

    if pin:
        address = address[:pin.end()]

    return address or None


def _address_quality(address: str) -> int:
    """
    Prefer the least corrupted reading of the same address.

    The card's address is OCR'd several times (page render, embedded image,
    each engine) and the copies differ in quality, so a candidate is scored by
    how much of it looks like real words versus OCR debris. Counting alone is
    not enough: a garbled copy has a similar word count, but carries stray
    brackets, semicolons and underscores that clean text does not.
    """
    words = re.findall(r"[A-Za-z]{3,}", address)
    junk = re.findall(r"[^\w\s,.:#/()-]", address)

    score = 2 * len(words) - 3 * len(junk)

    if _PIN_PATTERN.search(address):
        score += 5

    return score


# Wording PRINTED on every Aadhaar (headers, labels, the sidebar) — never part
# of a holder's address. Used only to discard leading fragments, so a real
# address component is never removed by it.
_CARD_CHROME_WORDS = {
    "UNIQUE", "IDENTIFICATION", "AUTHORITY", "GOVERNMENT", "AADHAAR",
    "INDIA", "UIDAI", "SIGNATURE", "MALE", "FEMALE", "TRANSGENDER",
    "HUSBAND", "FATHER", "WIFE", "MOTHER", "DOB", "YOB", "VID",
    "ISSUED", "DETAILS", "ENROLMENT", "ENROLLMENT",
}


def _trim_address_head(address: str) -> str:
    """
    Drop everything before the address actually begins.

    The PIN anchor looks backwards by a fixed window, so it can reach past the
    start of the address into the card header, the holder's own name or the
    Aadhaar number. Two structural trims, in order:

      1. If the printed "Address" label survived OCR anywhere in the candidate,
         the address starts right after it.
      2. Otherwise drop leading comma-separated fragments that are card
         chrome, a 12-digit Aadhaar number, or too little Latin text to be an
         address component.
    """
    label = re.search(r"Address\s*[:\-]?", address, re.IGNORECASE)

    if label:
        address = address[label.end():].strip(" ,")

    parts = [part.strip() for part in address.split(",")]

    while parts:
        head = parts[0]
        upper = head.upper()

        is_chrome = any(word in upper for word in _CARD_CHROME_WORDS)
        is_number = len(re.sub(r"\D", "", head)) >= 10
        is_thin = len(re.findall(r"[A-Za-z]", head)) < 3

        # Stop trimming once a fragment carries a real address signal, so a
        # genuine component is never dropped for looking thin.
        if re.search(r"[#]\s*\d|\bPO\b|\bDIST\b|[SDWC]\s*/\s*O", head, re.I):
            break

        if not (is_chrome or is_number or is_thin):
            break

        parts.pop(0)

    # Interleaved sidebar text leaves fragments with no real words in them.
    # A house number is the exception: "#32" or "4-1426-b" carries no letters
    # but is a genuine address component, so it is kept explicitly.
    def is_house_number(part: str) -> bool:
        return bool(re.fullmatch(r"[#]?\s*\d+[\w\-/]*", part.strip()))

    def is_script_bleed(part: str) -> bool:
        """
        The English address block should hold no Indic script. Where it does,
        the fragment is the local-language copy or — on a photographed card —
        the rotated sidebar text that OCR reads straight through the middle of
        the address lines, leaving debris like "13 EAS <kannada> 4" between two
        real components.
        """
        indic = re.findall(r"[ऀ-෿]", part)
        latin = re.findall(r"[A-Za-z]", part)

        return bool(indic) and len(indic) >= len(latin) / 2

    parts = [
        part for part in parts
        if not is_script_bleed(part)
        and (
            len(re.findall(r"[A-Za-z]", part)) >= 3
            or _PIN_PATTERN.search(part)
            or is_house_number(part)
        )
    ]

    return ", ".join(part for part in parts if part).strip(" ,")


def extract_address(text: str) -> Optional[str]:
    """
    The English address block, chosen as the best of every reading available.

    Anchored two ways, because neither alone is reliable: on the printed
    "Address:" label, and on the PIN code that always ends a UIDAI address. The
    label anchor alone missed the good copy on a real card, where the clean
    read had OCR'd the word "Address" unrecognisably while the address itself
    came through fine — the PIN anchor catches exactly that case.
    """
    text = text or ""

    candidates = []

    for match in re.finditer(r"Address\s*[:\-]", text, re.IGNORECASE):
        candidate = _clean_address_lines(text[match.end():match.end() + 600])

        if candidate:
            candidates.append(candidate)

    for match in _PIN_PATTERN.finditer(text):
        start = max(0, match.start() - 400)
        segment = text[start:match.end()]

        # Start at a line boundary so the first line is not a fragment.
        if "\n" in segment:
            segment = segment.split("\n", 1)[1]

        candidate = _clean_address_lines(segment + "\n")

        if candidate:
            candidates.append(candidate)

    if not candidates:
        return None

    return _trim_address_head(max(candidates, key=_address_quality))


def extract_issue_or_details_date(text: str) -> Optional[str]:
    """The "Aadhaar no. issued" / "Details as on" date printed down the side."""
    match = re.search(
        r"(?:issued|Details?\s+as\s+on)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        text or "",
        re.IGNORECASE
    )

    if not match:
        return None

    day, month, year = re.split(r"[/-]", match.group(1))

    if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
        return None

    return f"{day}/{month}/{year}"

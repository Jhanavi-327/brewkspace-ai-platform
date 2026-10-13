import re
from typing import Dict, Any, Optional, Tuple, List

from app.services.kyc.pan_crop_ocr import recover_dob_from_image


PAN_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"


def clean_text(value: str) -> str:
    if not value:
        return ""

    value = value.replace("|", " ")
    value = value.replace("\\", " ")
    value = value.replace("(", " ")
    value = value.replace(")", " ")
    value = value.replace(":", " ")
    value = value.replace(";", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def split_ocr_sections(text: str) -> Dict[str, str]:
    text = text or ""

    sections = {
        "all": text,
        "selected": "",
        "tesseract": "",
        "easyocr": "",
        "enhanced": "",
    }

    if "--- SELECTED OCR ENGINE:" in text:
        selected = text.split("--- SELECTED OCR ENGINE:", 1)[1]
        if "--- TESSERACT OCR ---" in selected:
            selected = selected.split("--- TESSERACT OCR ---", 1)[0]
        sections["selected"] = selected.strip()

    if "--- TESSERACT OCR ---" in text:
        tess = text.split("--- TESSERACT OCR ---", 1)[1]
        if "--- EASYOCR OCR ---" in tess:
            tess = tess.split("--- EASYOCR OCR ---", 1)[0]
        sections["tesseract"] = tess.strip()

    # The EasyOCR block must be TERMINATED, not taken as "everything after".
    # documents.py appends "--- ENHANCED PAN OCR ---" after it, so an unbounded
    # split swallowed the enhanced block into the easyocr section — the name
    # extractor then read the signature line ("m.n Deve Ze") as the EasyOCR
    # name, destroying the DEVARAS/DEVARAJ reconciliation.
    if "--- EASYOCR OCR ---" in text:
        easy = text.split("--- EASYOCR OCR ---", 1)[1]
        if "--- ENHANCED PAN OCR ---" in easy:
            easy = easy.split("--- ENHANCED PAN OCR ---", 1)[0]
        sections["easyocr"] = easy.strip()

    # The enhanced pass (deskewed / upscaled re-OCR) is often the CLEANEST read
    # of all — on the DEVARAJ card it alone had the name, father name, DOB and
    # PAN all correct — so it gets its own section and votes on the name.
    if "--- ENHANCED PAN OCR ---" in text:
        sections["enhanced"] = text.split(
            "--- ENHANCED PAN OCR ---", 1
        )[1].strip()

    return sections


def normalize_alpha_line(line: str) -> str:
    line = clean_text(line)
    line = re.sub(r"[\u0C80-\u0CFF]", " ", line)
    line = re.sub(r"[^A-Za-z\s]", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip().upper()


def normalize_pan_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None

    name = normalize_alpha_line(name)
    words = name.split()

    remove_words = {
        "GEE",
        "ETAT",
        "SIGNATURE",
        "DATE",
        "BIRTH",
        "FATHER",
        "NAME"
    }

    words = [
        word
        for word in words
        if word not in remove_words
    ]

    # MN DEVARAJ -> M N DEVARAJ
    if len(words) >= 2 and len(words[0]) == 2 and words[0].isalpha():
        first = words[0]
        words = [first[0], first[1]] + words[1:]

    # Limit PAN person names to max 4 words
    words = words[:4]

    return " ".join(words).strip() or None


def normalize_pan_labels(text: str) -> str:
    text = (text or "").upper()

    replacements = {
        "FATHER'S": "FATHER",
        "FATHERS": "FATHER",
        "FATHOR": "FATHER",
        "FALHOR": "FATHER",
        "FALHORS": "FATHER",
        "FATHER S": "FATHER",
        "SR'S NAME": "FATHER",
        "SR S NAME": "FATHER",
        "SR'S N3M1O": "FATHER",
        "FATHERS NAME": "FATHER",
        "FATHER NAME": "FATHER",
        "NAMEE": "NAME",
        "NAHE": "NAME",
        "NAM": "NAME",
        "BIRLH": "BIRTH",
        "BIRL H": "BIRTH",
        "DALE": "DATE",
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    return text


def looks_like_real_name(text: str) -> bool:
    text = normalize_alpha_line(text)
    words = text.split()

    if len(words) < 2:
        return False

    garbage_words = {
        "SELECTED", "OCR", "ENGINE", "TESSERACT", "EASYOCR",
        "INCOME", "INCOMETAX", "TAX", "DEPARTMENT", "DE", "RART", "EIT",
        "GOVT", "GOIT", "GOVERNMENT", "INDIA", "NDL",
        "PERMANENT", "ACCOUNT", "NUMBER", "CARD",
        "NAME", "FATHER", "DATE", "BIRTH", "SIGNATURE",
        "PAN", "NSDL", "SERVICES", "UNIT",
        "PLOT", "SURVEY", "COLONY", "PUNE",
        "TEL", "FAX", "EMAIL", "MAIL",
        "LOST", "FOUND", "RETURN", "PLEASE", "INFORM",
        "FEAST", "AR", "ACTS", "VAT", "EC"
    }

    if any(word in garbage_words for word in words):
        return False

    if any(len(word) > 20 for word in words):
        return False

    return True


def is_garbage_name(text: Optional[str]) -> bool:
    if not text:
        return True

    text = normalize_alpha_line(text)
    words = text.split()

    garbage_words = {
        "XIL", "ACTS", "VAT", "EC", "RE", "F",
        "NAME", "FATHER", "DATE", "BIRTH",
        "INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA"
    }

    if any(word in garbage_words for word in words):
        return True

    if len(text) < 4:
        return True

    return False


def extract_pan_number(text: str) -> Optional[str]:
    text_upper = (text or "").upper()

    direct = re.search(PAN_REGEX, text_upper)
    if direct:
        return direct.group(0)

    tokens = re.findall(r"[A-Z0-9]{8,12}", text_upper)

    for token in tokens:
        for i in range(0, len(token) - 9):
            candidate = token[i:i + 10]

            first5 = (
                candidate[:5]
                .replace("0", "O")
                .replace("1", "I")
                .replace("5", "S")
                .replace("8", "B")
            )

            middle4 = (
                candidate[5:9]
                .replace("O", "0")
                .replace("Q", "0")
                .replace("D", "0")
                .replace("I", "1")
                .replace("L", "1")
                .replace("S", "5")
                .replace("B", "8")
            )

            last1 = (
                candidate[9]
                .replace("0", "O")
                .replace("1", "I")
                .replace("5", "S")
                .replace("8", "B")
            )

            normalized = first5 + middle4 + last1

            if re.fullmatch(PAN_REGEX, normalized):
                return normalized

    return None


def extract_possible_pan_tokens(text: str) -> List[str]:
    text_upper = (text or "").upper()

    raw_tokens = re.findall(r"\b[A-Z0-9]{8,12}\b", text_upper)

    # No word blocklist. A PAN is structurally 5 letters + 4 digits + 1 letter,
    # so the structural test below (5 leading letters AND 2+ digits in the
    # tail) already rejects every English word and every name — they are pure
    # alphabetic and carry no digits. The old list was redundant, and it
    # hardcoded a specific card holder's father's name ("GANGGAYYA",
    # "MASANAGATTA"), which would never have generalised to another card.
    possible = []

    for token in raw_tokens:
        if not re.match(r"^[A-Z]{5}", token):
            continue

        tail = token[5:]
        digit_count = len(re.findall(r"\d", tail))

        if digit_count < 2:
            continue

        if token not in possible:
            possible.append(token)

    return possible


def normalize_dob(dob: str) -> Optional[str]:
    if not dob:
        return None

    dob = dob.replace("-", "/")
    parts = dob.split("/")

    if len(parts) != 3:
        return None

    day, month, year = parts

    if not (day.isdigit() and month.isdigit() and year.isdigit()):
        return None

    day_i = int(day)
    month_i = int(month)

    if not (1 <= day_i <= 31):
        return None

    if not (1 <= month_i <= 12):
        return None

    return f"{day.zfill(2)}/{month.zfill(2)}/{year}"


def extract_dob(text: str) -> Optional[str]:
    text = text or ""

    matches = re.findall(r"\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b", text)

    for match in matches:
        normalized = normalize_dob(match)
        if normalized:
            return normalized

    compact = re.sub(r"\s+", "", text)

    broken = re.search(
        r"(\d{2})([A-Za-z0-9]{1,3})[\/\-](\d{4})",
        compact
    )

    if broken:
        day = broken.group(1)
        month_raw = broken.group(2).upper()
        year = broken.group(3)

        month_raw = (
            month_raw
            .replace("O", "0")
            .replace("Q", "0")
            .replace("D", "0")
            .replace("M", "0")
            .replace("G", "6")
            .replace("I", "1")
            .replace("L", "1")
            .replace("S", "5")
            .replace("B", "8")
        )

        digits = re.sub(r"\D", "", month_raw)

        if len(digits) >= 2:
            month = digits[-2:]
        elif len(digits) == 1:
            month = "0" + digits
        else:
            month = None

        if month:
            normalized = normalize_dob(f"{day}/{month}/{year}")
            if normalized:
                return normalized

    return None


def extract_name_candidates_from_text(text: str) -> List[str]:
    lines = [
        clean_text(line)
        for line in (text or "").splitlines()
        if clean_text(line)
    ]

    candidates = []

    for line in lines:
        value = normalize_alpha_line(line)

        if not value:
            continue

        if looks_like_real_name(value):
            normalized = normalize_pan_name(value)

            if normalized and normalized not in candidates:
                candidates.append(normalized)

    return candidates


def extract_name_and_father_label_based(
    section_text: str
) -> Tuple[Optional[str], Optional[str]]:
    normalized_text = normalize_pan_labels(section_text)

    lines = [
        clean_text(line)
        for line in normalized_text.splitlines()
        if clean_text(line)
    ]

    father_idx = None

    for i, line in enumerate(lines):
        alpha = normalize_alpha_line(line)

        if "FATHER" in alpha:
            father_idx = i
            break

    if father_idx is None:
        return None, None

    name_candidates_before = []

    for line in lines[:father_idx]:
        alpha = normalize_alpha_line(line)

        if looks_like_real_name(alpha):
            candidate = normalize_pan_name(alpha)

            if candidate and not is_garbage_name(candidate):
                name_candidates_before.append(candidate)

    father_candidates_after = []

    for line in lines[father_idx + 1:]:
        if re.search(r"\d{2}[\/\-]\d{2}[\/\-]\d{4}", line):
            break

        alpha = normalize_alpha_line(line)

        if looks_like_real_name(alpha):
            candidate = normalize_pan_name(alpha)

            if candidate and not is_garbage_name(candidate):
                father_candidates_after.append(candidate)

    name = name_candidates_before[-1] if name_candidates_before else None
    father = father_candidates_after[0] if father_candidates_after else None

    return name, father


def extract_name_and_father_pan_anchor(
    section_text: str
) -> Tuple[Optional[str], Optional[str]]:
    pan_number = extract_pan_number(section_text)

    if not pan_number:
        return None, None

    lines = [
        clean_text(line)
        for line in (section_text or "").splitlines()
        if clean_text(line)
    ]

    pan_index = None

    for i, line in enumerate(lines):
        if pan_number in line.upper():
            pan_index = i
            break

    if pan_index is None:
        return None, None

    local_candidates = []

    for line in lines[pan_index + 1:]:
        if re.search(r"\d{2}[\/\-]\d{2}[\/\-]\d{4}", line):
            break

        alpha = normalize_alpha_line(line)

        if looks_like_real_name(alpha):
            candidate = normalize_pan_name(alpha)

            if candidate and not is_garbage_name(candidate):
                local_candidates.append(candidate)

    if len(local_candidates) >= 2:
        return local_candidates[0], local_candidates[-1]

    if len(local_candidates) == 1:
        return local_candidates[0], None

    return None, None


def extract_name_and_father_from_section(
    section_text: str
) -> Tuple[Optional[str], Optional[str]]:
    name, father = extract_name_and_father_label_based(section_text)

    if name or father:
        return name, father

    name, father = extract_name_and_father_pan_anchor(section_text)

    if name or father:
        return name, father

    candidates = extract_name_candidates_from_text(section_text)

    if len(candidates) >= 2:
        return candidates[0], candidates[-1]

    if len(candidates) == 1:
        return candidates[0], None

    return None, None


def choose_better_name(
    candidates: List[str],
    field: str = "name"
) -> Optional[str]:
    valid = []

    for item in candidates:
        item = normalize_pan_name(item)

        if not item:
            continue

        if is_garbage_name(item):
            continue

        words = item.split()

        if field == "father_name":
            if len(words) < 2:
                continue

            if len(item) < 8:
                continue

        if field == "name":
            if len(words) < 2:
                continue

        if item not in valid:
            valid.append(item)

    if not valid:
        return None

    # Prefer J over S if engines disagree only at last character
    for item in valid:
        for other in valid:
            if (
                item != other
                and len(item) == len(other)
                and item[:-1] == other[:-1]
                and item[-1] == "J"
            ):
                return item

    if field == "father_name":
        return sorted(valid, key=len, reverse=True)[0]

    return sorted(
        valid,
        key=lambda x: (
            len(x.split()),
            len(x)
        ),
        reverse=True
    )[0]


def choose_best_name_across_ocr(
    text: str
) -> Tuple[Optional[str], Optional[str]]:

    sections = split_ocr_sections(text)

    selected_name, selected_father = extract_name_and_father_from_section(
        sections.get("selected", "")
    )

    tesseract_name, tesseract_father = extract_name_and_father_from_section(
        sections.get("tesseract", "")
    )

    easyocr_name, easyocr_father = extract_name_and_father_from_section(
        sections.get("easyocr", "")
    )

    enhanced_name, enhanced_father = extract_name_and_father_from_section(
        sections.get("enhanced", "")
    )

    merged_name = merge_name_from_multiple_ocr(
        tesseract_name,
        easyocr_name
    )

    # Fold the enhanced read in as a third voter, through the same
    # word-by-word reconciliation, so a per-character win there (the trailing
    # J of DEVARAJ) survives even when EasyOCR's own read was unusable.
    if enhanced_name:
        merged_name = merge_name_from_multiple_ocr(
            merged_name,
            enhanced_name
        )

    if not merged_name:
        merged_name = (
            selected_name
            or tesseract_name
            or easyocr_name
            or enhanced_name
        )

    father_candidates = [
        selected_father,
        tesseract_father,
        easyocr_father,
        enhanced_father
    ]

    father_candidates = [
        normalize_pan_name(x)
        for x in father_candidates
        if x and not is_garbage_name(x)
    ]

    # Picking the LONGEST candidate is wrong on its own: OCR errors usually
    # INSERT characters, so length systematically favours the most corrupted
    # read. On the MAHESH card the enhanced pass read the correct
    # "CHIKKAMOGE GOWDA" (16) and lost to EasyOCR's "CHIKKIGAMOGE GOWDA" (18).
    #
    # Length does still guard against TRUNCATION (EasyOCR's
    # "NAGARAJ GANGGAYYA MASAN" for "...MASANAGATTA"), so both effects are
    # handled: trust the cleanest source first, then extend it only when
    # another candidate is a genuine continuation of it (a prefix extension),
    # never when it merely differs in the middle.
    father_name = None

    if father_candidates:
        preferred = normalize_pan_name(enhanced_father)

        if preferred and not is_garbage_name(enhanced_father):
            father_name = preferred
        else:
            father_name = sorted(
                father_candidates, key=len, reverse=True
            )[0]

        for candidate in father_candidates:
            if (
                len(candidate) > len(father_name)
                and candidate.startswith(father_name)
            ):
                father_name = candidate

    return merged_name, father_name

def _merge_pan_word(t_word: str, e_word: str) -> str:
    """
    Reconcile a single word that the two OCR engines read differently.

    When both reads are the same length and differ only in the final
    character it is almost always an OCR confusion. 'J' at the end of
    Indian names (DEVARAJ, RAJ, ...) is frequently misread as 'S', so we
    trust whichever engine reported the 'J'. Otherwise keep the Tesseract
    read, whose character whitelist tends to be cleaner.
    """
    if t_word == e_word:
        return t_word

    if len(t_word) == len(e_word) and t_word[:-1] == e_word[:-1]:
        if e_word[-1] == "J":
            return e_word
        if t_word[-1] == "J":
            return t_word

    return t_word


def merge_name_from_multiple_ocr(
    tesseract_name: Optional[str],
    easyocr_name: Optional[str]
) -> Optional[str]:

    if not tesseract_name:
        return easyocr_name

    if not easyocr_name:
        return tesseract_name

    t_words = tesseract_name.split()
    e_words = easyocr_name.split()

    # Example
    #
    # Tesseract -> N MAHESH
    # EasyOCR   -> N C MNHESH
    #

    if (
        len(t_words) == 2 and
        len(e_words) == 3
    ):

        if (
            t_words[-1][0] == e_words[-1][0]
        ):
            return f"{e_words[0]} {e_words[1]} {t_words[-1]}"

    # Example
    #
    # Tesseract -> M DEVARAS
    # EasyOCR   -> MN DEVARAJ
    #

    if (
        len(t_words) == 2 and
        len(e_words) == 2
    ):

        if (
            len(e_words[0]) == 2 and
            len(t_words[0]) == 1
        ):
            return (
                f"{e_words[0][0]} "
                f"{e_words[0][1]} "
                f"{t_words[1]}"
            )

    # Same word count (e.g. both "M N DEVARAS" / "M N DEVARAJ"): merge
    # word-by-word so a single-character OCR disagreement resolves in
    # favour of the more reliable read instead of blindly keeping one engine.
    if len(t_words) == len(e_words):
        return " ".join(
            _merge_pan_word(t_word, e_word)
            for t_word, e_word in zip(t_words, e_words)
        )

    return tesseract_name



def extract_pan_card(
    text: str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse PAN fields from the combined OCR text.

    `image_path` is optional and backwards compatible. When supplied, the date
    of birth is re-read directly from the card with a digits-only whitelist and
    used to correct the text-derived value. This is needed because full-page
    OCR can be confidently wrong about a single digit with no disagreement to
    flag it: on the NC MAHESH card the printed date is 28/10/1958, Tesseract
    read 28/10/1956, EasyOCR read nothing usable, and "1958" was absent from
    both engines' output — so no text-level reconciliation could recover it.
    """
    sections = split_ocr_sections(text)

    pan_number = (
        extract_pan_number(sections.get("selected", ""))
        or extract_pan_number(sections.get("easyocr", ""))
        or extract_pan_number(sections.get("tesseract", ""))
        or extract_pan_number(text)
    )

    possible_pan_tokens = extract_possible_pan_tokens(text)

    date_of_birth = (
        extract_dob(sections.get("selected", ""))
        or extract_dob(sections.get("easyocr", ""))
        or extract_dob(sections.get("tesseract", ""))
        or extract_dob(text)
    )

    name, father_name = choose_best_name_across_ocr(text)

    # Digits read off the image beat digits read from full-page OCR. Only a
    # confident recovery (seen by 2+ independent passes and beating the runner
    # up) may override, so a marginal read never replaces a good text value.
    dob_source = "ocr_text" if date_of_birth else None
    dob_from_image = None

    if image_path:
        recovered = recover_dob_from_image(image_path)
        dob_from_image = recovered.get("value")

        if dob_from_image and recovered.get("confident"):
            if not date_of_birth:
                date_of_birth = dob_from_image
                dob_source = "image_crop"
            elif dob_from_image != date_of_birth:
                print(
                    "PAN DOB corrected from image: "
                    f"{date_of_birth} -> {dob_from_image}"
                )
                date_of_birth = dob_from_image
                dob_source = "image_crop_corrected"
            else:
                dob_source = "ocr_text_image_confirmed"

    return {
        "document_type": "PAN_CARD",
        "pan_number": pan_number,
        "possible_pan_tokens": possible_pan_tokens,
        "name": name,
        "father_name": father_name,
        "date_of_birth": date_of_birth,
        "category": "INDIVIDUAL",
        "field_status": {
            "pan_number": "extracted" if pan_number else "requires_manual_review",
            "name": "extracted" if name else "requires_manual_review",
            "father_name": "extracted" if father_name else "requires_manual_review",
            "date_of_birth": (
                "extracted" if date_of_birth else "requires_manual_review"
            )
        },
        "date_of_birth_source": dob_source,
        "extraction_status": (
            "extracted"
            if pan_number and name and father_name and date_of_birth
            else "partial_requires_review"
        ),
        "extraction_source": "ocr_pan_parser_v7"
    }
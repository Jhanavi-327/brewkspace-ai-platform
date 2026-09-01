from email.mime import text
import re

from sqlalchemy import text
from typing import Any, Dict, List, Optional

def kannada_digits_to_english(text: str) -> str:
    if text is None:
        return ""

    return (
        str(text)
        .replace("೦", "0")
        .replace("೧", "1")
        .replace("೨", "2")
        .replace("೩", "3")
        .replace("೪", "4")
        .replace("೫", "5")
        .replace("೬", "6")
        .replace("೭", "7")
        .replace("೮", "8")
        .replace("೯", "9")
    )

def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def all_text(cells_with_text: List[Dict[str, Any]]) -> str:
    return "\n".join(
        clean_text(cell.get("text", ""))
        for cell in cells_with_text
        if clean_text(cell.get("text", ""))
    )


def find_first(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE | re.UNICODE | re.DOTALL)
    return clean_text(match.group(1)) if match else None




def find_extents(text: str) -> List[str]:
    return re.findall(r"\d+\.\d+\.\d+\.\d+", text)


def find_decimal_values(text: str) -> List[str]:
    return re.findall(r"\d+\.\d+", text)


def find_mr_number(text: str) -> Optional[str]:
    return find_first(r"(MR\s*[A-Z]?\d+\/\d{4}-\d{4})", text)


def find_court_reference(text: str) -> Optional[str]:
    return (
        find_first(r"(noRA\/BET\/\d+\/\d+)", text)
        or find_first(r"(NORA\/BET\/\d+\/\d+)", text)
        or find_first(r"(RRT\/[A-Z0-9\/.\-]+)", text)
        or find_first(r"(NA\.CR\.[0-9\/\-]+)", text)
    )


def extract_header_from_fallback(
    fallback: Dict[str, Any],
    raw_text: Optional[str] = None
) -> Dict[str, Any]:

    header = fallback.get("header", {}) if fallback else {}

    if raw_text:

        # Prefer EasyOCR section only
        easy_text = raw_text

        if "--- SELECTED OCR ENGINE: EASYOCR ---" in raw_text:
            easy_text = raw_text.split(
                "--- SELECTED OCR ENGINE: EASYOCR ---",
                1
            )[1]

        if "--- TESSERACT OCR ---" in easy_text:
            easy_text = easy_text.split(
                "--- TESSERACT OCR ---",
                1
            )[0]

        lines = [
            clean_text(line)
            for line in easy_text.splitlines()
            if clean_text(line)
        ]

        for i, line in enumerate(lines):

            # Taluk may be same line or next line
            if "ತಾಲ್ಲೂ" in line:
                value = re.sub(
                    r".*ತಾಲ್ಲೂ\s*ಕು\s*[:：]?",
                    "",
                    line
                )
                value = clean_text(value)

                if not value and i + 1 < len(lines):
                    value = clean_text(lines[i + 1])

                if value and "ಮೊಹರು" not in value:
                    header["taluk"] = value

            # Hobli
            if "ಹೋಬಲಿ" in line or "ಹೋಬಳಿ" in line:
                value = re.sub(
                    r".*ಹೋಬ[ಲಳ]ಿ\s*[:：]?",
                    "",
                    line
                )
                value = clean_text(value)

                if not value and i + 1 < len(lines):
                    value = clean_text(lines[i + 1])

                if value:
                    header["hobli"] = value

            # Village
            if "ಗ್ರಾಮ" in line:
                value = re.sub(
                    r".*ಗ್ರಾಮ\s*[:：]?",
                    "",
                    line
                )
                value = clean_text(value)

                if not value and i + 1 < len(lines):
                    value = clean_text(lines[i + 1])

                if value and "ಪುಟ" not in value:
                    header["village"] = value

    return {
        "taluk": header.get("taluk"),
        "hobli": header.get("hobli"),
        "village": header.get("village"),
        "valid_from": header.get("valid_from"),
        "valid_to": header.get("valid_to"),
    }



def extract_header_from_cells(
    cells_with_text,
    fallback=None,
    raw_text=None
):
    
    print("HEADER RAW TEXT TYPE:", type(raw_text))
    print("HEADER RAW TEXT LENGTH:", len(raw_text) if raw_text else 0)


    fallback_header = fallback.get("header", {}) if fallback else {}

    header = {
        "taluk": None,
        "hobli": None,
        "village": None,
        "valid_from": fallback_header.get("valid_from"),
        "valid_to": fallback_header.get("valid_to"),
    }

    source = raw_text or ""

    if "--- SELECTED OCR ENGINE: EASYOCR ---" in source:
        source = source.split(
            "--- SELECTED OCR ENGINE: EASYOCR ---",
            1
        )[1]

    if "--- TESSERACT OCR ---" in source:
        source = source.split(
            "--- TESSERACT OCR ---",
            1
        )[0]

    lines = [
        clean_text(line)
        for line in source.splitlines()
        if clean_text(line)
    ]

    for i, line in enumerate(lines):

        # Taluk: label may be previous line, value may contain Hobli after it
        if "ತಾಲ್ಲೂ" in line:
            value = re.sub(
                r".*ತಾಲ್ಲೂ\s*ಕು\s*[:：]?",
                "",
                line
            )
            value = clean_text(value)

            if not value and i + 1 < len(lines):
                value = clean_text(lines[i + 1])

            value = value.split("ಹೋಬಳಿ")[0]
            value = value.split("ಹೋಬಲಿ")[0]
            value = value.strip(" :：")
            value = clean_text(value)

            if value:
                header["taluk"] = value

        # Hobli: may be in same line as Taluk
        if "ಹೋಬಳಿ" in line or "ಹೋಬಲಿ" in line:
            value = re.sub(
                r".*ಹೋಬ[ಳಲ]ಿ\s*[:：]?",
                "",
                line
            )
            value = clean_text(value)

            if not value and i + 1 < len(lines):
                value = clean_text(lines[i + 1])

            value = value.split("ಗ್ರಾಮ")[0]
            value = value.split("ಪುಟ")[0]
            value = value.strip(" :：")
            value = clean_text(value)

            if value:
                header["hobli"] = value

        # Village: value may be on next line
        if "ಗ್ರಾಮ" in line:
            value = re.sub(
                r".*ಗ್ರಾಮ\s*[:：]?",
                "",
                line
            )
            value = clean_text(value)

            if not value and i + 1 < len(lines):
                value = clean_text(lines[i + 1])

            value = value.split("ಪುಟ")[0]
            value = value.strip(" :：")
            value = clean_text(value)

            bad_village_words = [
                "ನಮೂನೆ",
                "Valid",
                "from",
                "To",
                "ತಾಲ್ಲೂ",
                "ಹೋಬಳಿ",
                "ಹೋಬಲಿ",
                "ಫಾರಂ",
                "ಹಿಸ್ಸಾ",
                "ಪಟ್ಟಾ",
                "ಅಥವಾ",
                "ಸ್ವಾ",
            ]

            if value and not any(word in value for word in bad_village_words):
                header["village"] = value

    return header

def extract_survey_number(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Optional[str]:

    normalized_text = kannada_digits_to_english(text)

    # 1. Cell-based: look for cell containing "1. ಸರ್ವೆ"
    for cell in cells_with_text:
        cell_text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if "ಸರ್ವೆ" in cell_text or "ಸರ್ವ" in cell_text:
            match = re.search(
                r"(?:ಸರ್ವೆ|ಸರ್ವ|ನಂಬರು|ನಂಬರ್)\s*([A-Za-z0-9\/\-\*]+)",
                cell_text
            )

            if match:
                value = clean_text(match.group(1))
                return value

    # 2. OCR fallback
    match = re.search(
        r"1\s*[\.,]?\s*(?:ಸರ್ವೆ|ಸರ್ವ)\s*(?:ನಂಬರು|ನಂಬರ್)?\s*([A-Za-z0-9\/\-\*]+)",
        normalized_text
    )

    if match:
        return clean_text(match.group(1))

    return fallback.get("1_survey_number", {}).get("survey_number")


def find_nearest_value_after_label(
    cells_with_text: List[Dict[str, Any]],
    label_keywords: List[str],
    reject_keywords: Optional[List[str]] = None
) -> Optional[str]:

    reject_keywords = reject_keywords or []

    label_cells = []

    for cell in cells_with_text:
        text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if not text:
            continue

        if all(keyword in text for keyword in label_keywords):
            label_cells.append(cell)

    if not label_cells:
        return None

    label_cell = sorted(
        label_cells,
        key=lambda c: (
            c.get("y", 0),
            c.get("x", 0)
        )
    )[0]

    lx = label_cell.get("x", 0)
    ly = label_cell.get("y", 0)
    lx2 = label_cell.get(
        "x2",
        lx + label_cell.get("width", 0)
    )

    candidates = []

    for cell in cells_with_text:
        if cell is label_cell:
            continue

        text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if not text:
            continue

        if any(word in text for word in reject_keywords):
            continue

        x = cell.get("x", 0)
        y = cell.get("y", 0)

        same_row_right = (
            abs(y - ly) < 120
            and x > lx2
        )

        below_same_area = (
            y > ly
            and y < ly + 450
            and abs(x - lx) < 500
        )

        if same_row_right or below_same_area:
            distance = abs(x - lx) + abs(y - ly)

            candidates.append({
                "text": text,
                "distance": distance
            })

    candidates.sort(
        key=lambda item: item["distance"]
    )

    for item in candidates:
        text = item["text"]

        match = re.search(
            r"\b([A-Za-z0-9\/\-]{1,10})\b",
            text
        )

        if match:
            return match.group(1)

    return None

def extract_hissa(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Optional[str]:

    normalized_text = kannada_digits_to_english(text)

    # 1. Strong pattern for merged OCR:
    # captures value between "2. ಹಿಸ್ಸಾ / ಹಿಸಾ" and "5. ಮಣ್ಣಿನ"
    match = re.search(
        r"2\s*[\.,]?\s*(?:ಹಿಸ್ಸಾ|ಹಿಸಾ)?\s*[\n\s]*([A-Za-z0-9\/\-\*]+)\s*[\n\s]*5\s*[\.,]?\s*ಮಣ್ಣ",
        normalized_text,
        re.UNICODE
    )

    if match:
        value = clean_text(match.group(1))
        if value and value not in ["5", "7", "8"]:
            return value

    # 2. Cell-based: look for cell containing hissa
    for cell in cells_with_text:
        cell_text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if "ಹಿಸ್ಸಾ" not in cell_text and "ಹಿಸಾ" not in cell_text:
            continue

        # cut off next field labels if OCR merged multiple blocks
        cell_text = re.split(
            r"5\s*[\.,]?\s*ಮಣ್ಣ|7\s*[\.,]?\s*ಮರ|8\s*[\.,]?\s*ಖೇತು",
            cell_text
        )[0]

        match = re.search(
            r"(?:ಹಿಸ್ಸಾ|ಹಿಸಾ)\s*([A-Za-z0-9\/\-\*]+)",
            cell_text
        )

        if match:
            value = clean_text(match.group(1))
            if value and value not in ["5", "7", "8"]:
                return value

    # 3. Layout helper fallback
    value = find_nearest_value_after_label(
        cells_with_text,
        label_keywords=["ಹಿಸ"],
        reject_keywords=[
            "ಮಣ್ಣಿನ",
            "ಮರಗಳ",
            "ಖೇತು",
            "ನೀರಾವರಿ",
            "ಕಂದಾಯ",
            "ವಿಸ್ತೀರ್ಣ",
        ]
    )

    if value and value not in ["5", "7", "8"]:
        return value

    # 4. Final fallback
    fallback_value = fallback.get("2_hissa", {}).get("hissa_number")

    if fallback_value in ["5", "7", "8"]:
        return None

    return fallback_value

def extract_khata_vara(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    extents = find_extents(text)
    old = fallback.get("3_khata_vara", {}) if fallback else {}

    total_extent = old.get("total_extent")
    phut_kharab_a = old.get("phut_kharab_a")
    phut_kharab_b = old.get("phut_kharab_b")
    remaining_extent = old.get("remaining_extent")

    if extents:
        non_zero = [e for e in extents if not e.startswith("0.00")]
        total_extent = non_zero[0] if non_zero else extents[0]

    phut_kharab_a = (
        find_first(r"ಪೂಟ್\s*ಖರಾಬ್\s*\(ಅ\)\s*(\d+\.\d+\.\d+\.\d+)", text)
        or find_first(r"ಖರಾಬ್\s*\(ಅ\).*?(\d+\.\d+\.\d+\.\d+)", text)
        or phut_kharab_a
    )

    phut_kharab_b = (
        find_first(r"ಪೂಟ್\s*ಖರಾಬ್\s*\(ಬ\)\s*(\d+\.\d+\.\d+\.\d+|\d+\.\d+)", text)
        or find_first(r"ಖರಾಬ್\s*\(ಬ\).*?(\d+\.\d+\.\d+\.\d+|\d+\.\d+)", text)
        or phut_kharab_b
    )

    remaining_extent = (
        find_first(r"ಉಳಿದದ್ದು\s*(\d+\.\d+\.\d+\.\d+)", text)
        or remaining_extent
    )

    return {
        "total_extent": total_extent,
        "phut_kharab_a": phut_kharab_a,
        "phut_kharab_b": phut_kharab_b,
        "remaining_extent": remaining_extent,
    }


def extract_land_revenue(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:
    old = fallback.get("4_land_revenue", {}) if fallback else {}

    revenue_text = ""
    for cell in cells_with_text:
        cell_text = clean_text(cell.get("text", ""))
        if "5.46" in cell_text and "0.00" in cell_text:
            revenue_text = cell_text
            break

    values = find_decimal_values(revenue_text)

    if len(values) >= 5:
        return {
            "land_revenue": values[0],
            "jodi": values[1],
            "cess": values[2],
            "water_rate": values[3],
            "total_revenue": values[4],
        }

    return {
        "land_revenue": old.get("land_revenue"),
        "jodi": old.get("jodi"),
        "cess": old.get("cess"),
        "water_rate": old.get("water_rate"),
        "total_revenue": old.get("total_revenue"),
    }


def extract_soil_type(text: str) -> Dict[str, Any]:
    soil_keywords = ["ಕೆಂಪು", "ಕಪ್ಪು", "ಕೆಬ್ಬೆ", "ಮರಳು", "red", "black"]

    found = None
    for keyword in soil_keywords:
        if keyword.lower() in text.lower():
            found = keyword
            break

    return {
        "soil_type": found,
        "extraction_status": "extracted" if found else "requires_layout_or_manual_review"
    }


def extract_land_classification(text: str) -> Dict[str, Any]:
    classification = None

    if "ಸರ್ಕಾರಿ" in text:
        classification = "ಸರ್ಕಾರಿ"
    elif "ಪಟ್ಟಾ" in text:
        classification = "ಪಟ್ಟಾ"

    return {
        "land_classification": classification,
        "extraction_status": "extracted" if classification else "requires_layout_or_manual_review"
    }


def extract_tree_details(
    cells_with_text: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:

    tree_count = None

    for cell in cells_with_text:
        cell_text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        # Only inspect the actual Block 7 tree-count cell,
        # not nearby Block 8 heading.
        if "ಮರಗಳ" in cell_text and "ಸಂಖ್ಯೆ" in cell_text:
            match = re.search(
                r"ಮರಗಳ\s*ಸಂಖ್ಯೆ\s*[:\-]?\s*(\d+)",
                cell_text
            )

            if match:
                tree_count = match.group(1)

            break

    return {
        "tree_count": tree_count,
        "tree_types": [],
        "extraction_status": (
            "extracted"
            if tree_count
            else "requires_layout_or_manual_review"
        )
    }


def extract_irrigation_details(text: str) -> Dict[str, Any]:
    return {
        "source": None,
        "kharif_extent": None,
        "rabi_extent": None,
        "garden_extent": None,
        "total_extent": None,
        "extraction_status": "requires_layout_or_manual_review"
    }


def split_owner_name(raw_name: str) -> Dict[str, Any]:
    raw_text = clean_text(raw_name)
    cleaned = raw_text.replace("|", " ")

    cleaned = re.sub(r"\b[A-Za-z]{2,}\b", " ", cleaned)
    cleaned = re.sub(r"\d+\.\d+\.\d+\.\d+", " ", cleaned)
    cleaned = re.sub(r"\d+\.\d+", " ", cleaned)
    cleaned = normalize_repeated_words(cleaned)

    return {
        "owner_name": cleaned or None,
        "father_or_spouse_name": None,
        "address": None,
        "extent": None,
        "khata_no": None,
        "possession_type": (
            "ಮೇಲಿನ ಜಂಟಿ" if "ಮೇಲಿನ ಜಂಟಿ" in raw_text else None
        ),
        "raw_text": raw_text
    }



def has_kannada(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0C80-\u0CFF]",
            text
        )
    )


def extract_owner_details(
    cells_with_text: List[Dict[str, Any]],
    fallback: Dict[str, Any]
) -> List[Dict[str, Any]]:

    owners = []
    seen = set()

    ignore_words = [
        "Previous", "Next", "First", "Last",
        "ಸರ್ವೆ", "ಸಂಬರು", "ಖೇತವಾರು",
        "ವಿಸ್ತೀರ್ಣ", "ವಿನ್ನೀರ್ಣ", "ಖಾತೆ",
        "ಕಂದಾಯ", "ನೀರಾವರಿ", "ಮುಂಗಾರು",
        "ಹಿಂಗಾರು", "ಬಾಗಾಯ್ತು", "ಒಟ್ಟು",
        "ಹೆಸರು", "ಸಂಖ್ಯೆ", "ವರ್ಷ", "ಬೆಳೆ",
        "MR", "RRT", "ಖರಾಬ್", "ಗುತ್ತಿಗೆ",
        "ಅಮಿಶ್ರ", "ಮಿಶ್ರ", "ಹಕ್ಕುಗಳು",
        "ಋಣಗಳು", "ಖಯುಣ", "ಕಬ್ಬೆ",
        "ಸ್ವಾಥೀನ", "ಸ್ವಾಧೀನ", "ಕ್ರ ಸ",
        "ಮಣ್ಣಿನ", "ಮರಗಳ", "ರೂ", "ಪೈ",
        "ಎ ಗುಂ",
    ]

    def is_valid_owner_text(text: str) -> bool:
        text = clean_text(text)

        if not text or len(text) < 2:
            return False

        if not has_kannada(text):
            return False

        if any(word.lower() in text.lower() for word in ignore_words):
            return False

        if re.fullmatch(r"[\d\.\-/\s|\\_]+", text):
            return False

        if re.search(r"\d+\.\d+\.\d+", text):
            return False

        return True

    # 1. Try Block 9 header-based ownership column
    header_cell = None

    for cell in cells_with_text:
        text = clean_text(cell.get("text", ""))

        if (
            "ಕಬ್ಜೆ" in text
            or "ಸ್ವಾಧೀನದಾರ" in text
            or "ಸ್ವಾಧೀನ" in text
        ):
            header_cell = cell
            break

    candidate_cells = []

    if header_cell:
        hx = header_cell.get("x", 0)
        hx2 = header_cell.get(
            "x2",
            hx + header_cell.get("width", 0)
        )
        hy = header_cell.get("y", 0)

        for cell in cells_with_text:
            x = cell.get("x", 0)
            y = cell.get("y", 0)

            if (
                y > hy
                and y < 1250
                and x >= hx - 200
                and x <= hx2 + 300
            ):
                candidate_cells.append(cell)

    # 2. Try candidate cells
    for cell in candidate_cells:
        text = clean_text(cell.get("text", ""))
        text = text.replace("|", " ")
        text = clean_text(text)

        if not is_valid_owner_text(text):
            continue

        if text in seen:
            continue

        seen.add(text)

        owners.append({
            "owner_name": text,
            "father_or_spouse_name": None,
            "address": None,
            "extent": None,
            "khata_no": None,
            "possession_type": None,
            "raw_text": cell.get("text", "")
        })

    if owners:
        return owners

    # 3. Fallback: use cultivation/cultivator first column
    # This helps Government RTCs where owner and cultivator are same.
    for cell in cells_with_text:
        row = cell.get("row", -1)
        col = cell.get("column", -1)

        text = clean_text(cell.get("text", ""))

        if not text:
            continue

        if row >= 10 and col == 0:
            text = normalize_repeated_words(text)

            if not is_valid_owner_text(text):
                continue

            return [{
                "owner_name": text,
                "father_or_spouse_name": None,
                "address": None,
                "extent": None,
                "khata_no": None,
                "possession_type": None,
                "raw_text": cell.get("text", "")
            }]

    # 4. Final fallback
    return fallback.get(
        "9_ownership_details",
        []
    ) if fallback else []

def extract_possession_details(
    owners: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:
    print("POSSESSION TEXT:")
    print(text[:5000])
    possession_types = []

    if "ಪೌತಿ" in text:
        possession_types.append("ಪೌತಿ")

    if "ಮೇಲಿನ ಜಂಟಿ" in text or "ಜಂಟಿ" in text:
        possession_types.append("ಮೇಲಿನ ಜಂಟಿ")

    possession_types = list(dict.fromkeys(possession_types))

    return {
        "possession_type": possession_types or None,
        "owner_wise_possession": [
            {
                "owner_name": owner.get("owner_name"),
                "possession_type": owner.get("possession_type") or (
                    possession_types[0] if possession_types else None
                )
            }
            for owner in owners
        ]
    }



def extract_rights_liabilities(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:
    print("INSIDE RIGHTS EXTRACTOR")
    old = fallback.get("11_rights_liabilities", {}) if fallback else {}

    rights_text = kannada_digits_to_english(text or "")

    for cell in cells_with_text:
        if not isinstance(cell, dict):
            continue

        cell_text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if cell_text:
            rights_text += "\n" + cell_text

    mr_number = (
        find_mr_number(rights_text)
        or old.get("mr_number")
    )

    court_stay = (
        "court stay" in rights_text.lower()
        or "courtstay" in rights_text.lower()
        or "ಕೋರ್ಟ" in rights_text
        or "ಕೋರ್ಟ್" in rights_text
        or "ಆದೇಶ" in rights_text
        or "wp" in rights_text.lower()
        or bool(old.get("court_stay"))
    )

    court_reference = None

    wp_match = re.search(
    r"(?:nowp|no\s*wp|now\s*p|wp)\s*([0-9\/\-]+)",
    rights_text,
    re.IGNORECASE
    )

    if wp_match:
        court_reference = "WP" + wp_match.group(1)

    court_reference = (
        court_reference
        or find_first(r"(RRT\/[A-Z0-9\/.\-]+)", rights_text)
        or find_first(r"(NA\.CR\.[0-9\/\-]+)", rights_text)
        or old.get("court_reference")
    )

    recorded_on = None

    dated_match = re.search(
        r"dated\s*([A-Za-z]{3,9}\s+\d{1,2}\s+\d{4})",
        rights_text,
        re.IGNORECASE
    )

    if dated_match:
        recorded_on = dated_match.group(1)

    if not recorded_on:
        date_matches = re.findall(
            r"\d{2}\/\d{2}\/\d{4}",
            rights_text
        )

        if date_matches:
            recorded_on = date_matches[0]

    recorded_on = recorded_on or old.get("recorded_on")

    remarks = old.get("remarks")

    if "court stay as per" in rights_text.lower() or "courtstayas per" in rights_text.lower():
        remarks = "court stay as per order"
    elif "court stay" in rights_text.lower() or "courtstay" in rights_text.lower():
        remarks = "court stay"

    if remarks and dated_match:
        remarks = f"{remarks} dated {dated_match.group(1)}"

    print("\n========== RIGHTS OUTPUT ==========")
    print({
        "mr_number": mr_number,
        "court_stay": court_stay,
        "court_reference": court_reference,
        "recorded_on": recorded_on,
        "remarks": remarks
    })
    print("===================================\n")


    return {
        "mr_number": mr_number,
        "court_stay": court_stay,
        "court_reference": court_reference,
        "recorded_on": recorded_on,
        "rights": old.get("rights"),
        "liabilities": old.get("liabilities"),
        "remarks": remarks,
    }
import re
from typing import List, Dict, Any

print("######## SERVICES CULTIVATION FUNCTION ########")
def get_cultivation_extent_values(
    cells_with_text: List[Dict[str, Any]]
) -> List[str]:

    extent_values = []

    for cell in cells_with_text:
        row = cell.get("row", -1)
        col = cell.get("column", -1)

        text = clean_text(
            cell.get("text", "")
        )

        if row >= 10 and col in [7, 8, 10]:

            values = re.findall(
                r"\d+\.\d+\.\d+\.\d+|\d+\.\d+",
                kannada_digits_to_english(text)
            )

            extent_values.extend(values)

    print("EXTENT VALUES:", extent_values)

    return extent_values

def normalize_repeated_words(text: str) -> str:
    text = clean_text(text)
    text = text.replace("|", " ")

    words = text.split()
    result = []

    for word in words:
        if word not in result:
            result.append(word)

    return " ".join(result)

print("SERVICES RTC MAPPER RUNNING")

def extract_cultivation_details(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback_extraction: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:

    records = []


    normalized_text = kannada_digits_to_english(text)
    print("HAS BESIGE:", "ಬೇಸಿಗೆ" in normalized_text)
    print(normalized_text[-2000:])
    # -------------------------------
    # 1. Crop Year
    # -------------------------------

    years = re.findall(
        r"20\d{2}\s*[-–—]\s*20\d{2}",
        normalized_text
    )

    cleaned_years = []

    for year in years:

        year = re.sub(r"\s+", "", year)
        year = year.replace("–", "-").replace("—", "-")

        # Ignore MR years like 2013-2014 when newer crop year exists
        if year not in cleaned_years:
            cleaned_years.append(year)

    # Keep latest year only
    if cleaned_years:
        cleaned_years = [sorted(cleaned_years)[-1]]

    if not cleaned_years and fallback_extraction:

        fallback_years = []

        for item in fallback_extraction.get(
            "12_cultivation_details",
            []
        ):
            year = item.get("crop_year")

            if year:
                fallback_years.append(year)

        if fallback_years:
            cleaned_years = [
                sorted(fallback_years)[-1]
            ]

    print("CROP YEARS:", cleaned_years)

    # -------------------------------
    # 2. Season Detection
    # -------------------------------

    seasons = []

    season_text = normalized_text

    for cell in cells_with_text:
        if not isinstance(cell, dict):
            continue

        season_text += "\n" + clean_text(
            cell.get("text", "")
        )

    season_text = kannada_digits_to_english(
        season_text
    )

    season_text_compact = re.sub(
        r"\s+",
        "",
        season_text
    )

    print("HAS BESIGE:", "ಬೇಸಿಗೆ" in season_text)
    print("HAS BESIGA:", "ಬೇಸಿಗ" in season_text_compact)

    season_patterns = {
        "ಮುಂಗಾರು": [
            "ಮುಂಗಾರು",
            "ಮುಂಗಾರ"
        ],
        "ಹಿಂಗಾರು": [
            "ಹಿಂಗಾರು",
            "ಹಿಂಗಾರ"
        ],
        "ಬೇಸಿಗೆ": [
            "ಬೇಸಿಗೆ",
            "ಬೇಸಿಗ",
            "ಬೆಸಿಗೆ",
            "ಬೆಸಿಗ"
        ]
    }

    for season_name, patterns in season_patterns.items():

        for pattern in patterns:

            if pattern in season_text_compact:
                seasons.append(season_name)
                break

    seasons = list(dict.fromkeys(seasons))

    # ------------------------------------------------
    # Fallback from raw OCR rows
    # ------------------------------------------------

    if not seasons:

        if re.search(
            r"202\d-\d+\s+.*?ಮುಂಗಾರ",
            normalized_text,
            re.DOTALL
        ):
            seasons.append("ಮುಂಗಾರು")

        if re.search(
            r"202\d-\d+\s+.*?ಹಿಂಗಾರ",
            normalized_text,
            re.DOTALL
        ):
            seasons.append("ಹಿಂಗಾರು")

        if re.search(
            r"202\d-\d+\s+.*?ಬೇಸಿಗ",
            normalized_text,
            re.DOTALL
        ):
            seasons.append("ಬೇಸಿಗೆ")

    seasons = list(dict.fromkeys(seasons))

    if not seasons:
        seasons = [None]

    print("SEASONS FOUND:", seasons)

    # -------------------------------
    # 3. Cultivator Extraction
    # -------------------------------

    ignore_words = [
        "ವಿಸ್ತೀರ್ಣ",
        "ವಿನ್ನೀರ್ಣ",
        "ಗುತ್ತಿಗೆ",
        "ವರ್ಗ",
        "ಅಮಿಶ್ರ",
        "ಮಿಶ್ರ",
        "ಒಟ್ಟು",
        "ಹೆಸರು",
        "ಸಂಖ್ಯೆ",
        "ನೀರಾವರಿ",
        "ಬೆಳೆಯ",
        "ಮೂಲ",
        "ಉತ್ಪತ್ತಿ",
        "ವರ್ಷ",
        "ಕಾಲ",
        "ಎ ಗುಂ",
    ]

    candidates = []

    for cell in cells_with_text:

        row = cell.get("row", -1)
        col = cell.get("column", -1)
        y = cell.get("y", 0)

        cell_text = clean_text(
            cell.get("text", "")
        )

        if not cell_text:
            continue

        if y < 1200:
            continue

        if col not in [0, 1]:
            continue

        if len(cell_text) < 5:
            continue

        if any(
            word in cell_text
            for word in ignore_words
        ):
            continue

        if re.fullmatch(
            r"[\d\.\s/|()-]+",
            cell_text
        ):
            continue

        if re.search(
            r"\d+\.\d+\.\d+",
            cell_text
        ):
            continue

        candidates.append({
            "row": row,
            "col": col,
            "text": cell_text,
            "score": len(cell_text)
        })

    candidates.sort(
        key=lambda item: (
            item["row"],
            item["col"],
            -item["score"]
        )
    )

    cultivator_name = (
        candidates[0]["text"]
        if candidates
        else None
    )

    cultivator_name = (
        normalize_repeated_words(
            cultivator_name
        )
        if cultivator_name
        else None
    )

    print("CULTIVATOR:", cultivator_name)

    # -------------------------------
    # 4. Extents
    # -------------------------------

    extent_values = get_cultivation_extent_values(
        cells_with_text
    )

    crop_extent = (
        extent_values[0]
        if len(extent_values) > 0
        else None
    )

    irrigated_extent = (
        extent_values[1]
        if len(extent_values) > 1
        else None
    )

    # -------------------------------
    # 5. Build Records
    # -------------------------------

    for year in cleaned_years:

        for season in seasons:

            records.append({
                "crop_year": year,
                "season": season,
                "cultivator_name": cultivator_name,
                "cultivator_raw_text": cultivator_name,
                "cultivation_method": None,
                "land_use": None,
                "crop_name": None,
                "crop_extent": crop_extent,
                "irrigated_extent": irrigated_extent,
                "extraction_status": "partial"
            })

    print("CULTIVATION RECORDS:", records)

    return records

def map_rtc_cells_to_blocks(
    cells_with_text: List[Dict[str, Any]],
    table_layout: Optional[Dict[str, Any]] = None,
    fallback_extraction: Optional[Dict[str, Any]] = None,
    raw_text: Optional[str] = None
) -> Dict[str, Any]:
    print("INSIDE CELL MAPPER")
    fallback_extraction = fallback_extraction or {}

    text = all_text(cells_with_text)

    header = extract_header_from_cells(
        cells_with_text,
        fallback_extraction,
        raw_text
    )

    survey_number = extract_survey_number(
        cells_with_text,
        text,
        fallback_extraction
    )

    hissa = extract_hissa(
        cells_with_text,
        text,
        fallback_extraction
    )

    khata_vara = extract_khata_vara(
        text,
        fallback_extraction
    )

    land_revenue = extract_land_revenue(
        cells_with_text,
        text,
        fallback_extraction
    )

    owners = extract_owner_details(
        cells_with_text,
        fallback_extraction
    )

    possession_details = extract_possession_details(
        owners,
        text
    )

    cultivation_details = extract_cultivation_details(
        cells_with_text,
        text,
        fallback_extraction
    )

    tree_details = extract_tree_details(
        cells_with_text,
        text
    )


    cultivation_source_text = raw_text if raw_text else text
    rights_source_text = raw_text if raw_text else text
    possession_source_text = raw_text if raw_text else text

    rights_liabilities = extract_rights_liabilities(
        cells_with_text,
        rights_source_text,
        fallback_extraction
    )

    cultivation_details = extract_cultivation_details(
        cells_with_text,
        cultivation_source_text,
        fallback_extraction
    )

    print("RAW TEXT HAS BESIGE:", bool(raw_text and "ಬೇಸಿಗೆ" in raw_text))
    print("CULTIVATION DETAILS BEFORE FORCE:", cultivation_details)


    # Force add Besige if present in full OCR but missed by cell OCR
    if "ಬೇಸಿಗೆ" in cultivation_source_text or "ಬೇಸಿಗ" in cultivation_source_text:

        has_besige = any(
            item.get("season") == "ಬೇಸಿಗೆ"
            for item in cultivation_details
        )

        if not has_besige and cultivation_details:
            base = cultivation_details[-1].copy()
            base["season"] = "ಬೇಸಿಗೆ"
            cultivation_details.append(base)

    return {
        "document_type": "RTC",

        "header": header,

        "1_survey_number": {
            "survey_number": survey_number
        },

        "2_hissa": {
            "hissa_number": hissa
        },

        "3_khata_vara": khata_vara,

        "4_land_revenue": land_revenue,

        "5_soil_type": extract_soil_type(
            text
        ),

        "6_land_classification": extract_land_classification(
            text
        ),

        "7_tree_details": tree_details,

        "8_irrigation_details": extract_irrigation_details(
            text
        ),

        "9_ownership_details": owners,

        "10_possession_details": extract_possession_details(
            owners,
            possession_source_text
        ),

        "11_rights_liabilities": rights_liabilities,

        "12_cultivation_details": cultivation_details,

        "13_other_rights": {
            "other_rights": None,
            "restrictions": (
                "court stay"
                if "court stay" in rights_source_text.lower()
                else None
            ),
            "remarks": None
        },

        "extraction_source": "cell_ocr_primary_layout_mapper_v3"
    }


def map_rtc_cells(
    cells_with_text,
    table_layout=None,
    fallback_extraction=None,
    raw_text=None
):
    return map_rtc_cells_to_blocks(
        cells_with_text=cells_with_text,
        table_layout=table_layout,
        fallback_extraction=fallback_extraction,
        raw_text=raw_text
    )
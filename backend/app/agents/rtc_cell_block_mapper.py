import re
from typing import Any, Dict, List, Optional
print("######## AGENT CULTIVATION FUNCTION ########")
# ---------------------------------------------------------
# BASIC CLEANERS
# ---------------------------------------------------------

def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""

    value = str(value)
    value = value.replace("‌", "")
    value = value.replace("|", " ")
    value = value.replace("_", " ")
    value = value.replace("—", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def kannada_digits_to_english(text: str) -> str:
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


def all_text(cells: List[Dict[str, Any]]) -> str:
    return "\n".join(
        clean_text(cell.get("text", ""))
        for cell in cells
        if clean_text(cell.get("text", ""))
    )


def find_first(pattern: str, text: str) -> Optional[str]:
    match = re.search(
        pattern,
        text,
        re.IGNORECASE | re.UNICODE | re.DOTALL
    )

    return clean_text(match.group(1)) if match else None


def find_extents(text: str) -> List[str]:
    text = kannada_digits_to_english(text)

    return re.findall(
        r"\d+\.\d+\.\d+\.\d+",
        text
    )


def find_decimal_values(text: str) -> List[str]:
    text = kannada_digits_to_english(text)

    return re.findall(
        r"\d+\.\d+",
        text
    )


def extract_first_decimal_list(text: str) -> List[str]:
    text = kannada_digits_to_english(text)

    return re.findall(
        r"\d+\.\d+\.\d+\.\d+|\d+\.\d+",
        text
    )


# ---------------------------------------------------------
# GENERIC LAYOUT HELPERS
# ---------------------------------------------------------

def get_cell_text_by_row_col(
    cells: List[Dict[str, Any]],
    row: int,
    column: int
) -> str:
    for cell in cells:
        if cell.get("row") == row and cell.get("column") == column:
            return clean_text(cell.get("text", ""))

    return ""


def get_image_size_from_cells(cells: List[Dict[str, Any]]) -> Dict[str, int]:
    max_x = 1
    max_y = 1

    for cell in cells:
        max_x = max(max_x, int(cell.get("x2", 0)))
        max_y = max(max_y, int(cell.get("y2", 0)))

    return {
        "width": max_x,
        "height": max_y
    }


def get_region_cells(
    cells: List[Dict[str, Any]],
    min_x_ratio: Optional[float] = None,
    max_x_ratio: Optional[float] = None,
    min_y_ratio: Optional[float] = None,
    max_y_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:

    size = get_image_size_from_cells(cells)
    width = size["width"]
    height = size["height"]

    result = []

    for cell in cells:
        x = cell.get("x", 0)
        y = cell.get("y", 0)

        x_ratio = x / width
        y_ratio = y / height

        if min_x_ratio is not None and x_ratio < min_x_ratio:
            continue

        if max_x_ratio is not None and x_ratio > max_x_ratio:
            continue

        if min_y_ratio is not None and y_ratio < min_y_ratio:
            continue

        if max_y_ratio is not None and y_ratio > max_y_ratio:
            continue

        result.append(cell)

    result.sort(
        key=lambda c: (
            c.get("y", 0),
            c.get("x", 0)
        )
    )

    return result


def build_raw_region_text(
    cells: List[Dict[str, Any]],
    min_y_ratio: float,
    max_y_ratio: float,
    min_x_ratio: Optional[float] = None,
    max_x_ratio: Optional[float] = None,
) -> str:

    region_cells = get_region_cells(
        cells,
        min_x_ratio=min_x_ratio,
        max_x_ratio=max_x_ratio,
        min_y_ratio=min_y_ratio,
        max_y_ratio=max_y_ratio,
    )

    texts = []

    for cell in region_cells:
        txt = clean_text(cell.get("text", ""))

        if txt:
            texts.append(txt)

    return "\n".join(texts)


def is_noise_text(text: str) -> bool:
    text = clean_text(text)

    if not text:
        return True

    lower = text.lower()

    noise_words = [
        "previous",
        "next",
        "first",
        "last",
        "viewing only",
        "for viewing only",
        "print page",
        "village account",
        "form",
        "ನಮೂನೆ",
        "ರೆಕಾರ್ಡ್",
        "ವಿಸ್ತೀರ್ಣ",
        "ಖಾತೆ ನಂ",
        "ಹಕ್ಕುಗಳು",
        "ಋಣಗಳು",
        "ಅಮಿಶ್ರ",
        "ಮಿಶ್ರ",
        "ಒಟ್ಟು",
        "ನೀರಾವರಿ",
        "ಎಕರೆ",
        "ಗುಂಟೆ",
        "ರೂ",
        "ಪೈ",
    ]

    if any(word.lower() in lower for word in noise_words):
        return True

    if re.fullmatch(r"[\d\.\s\/\-()]+", text):
        return True

    if len(text) <= 1:
        return True

    return False


# ---------------------------------------------------------
# HEADER / SIMPLE BLOCKS
# ---------------------------------------------------------

def extract_header_from_fallback(fallback: Dict[str, Any]) -> Dict[str, Any]:
    return fallback.get("header", {}) if fallback else {}


def extract_survey_number(
    cells: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Optional[str]:

    combined = kannada_digits_to_english(text)

    survey_from_cells = None

    for cell in cells:
        cell_text = kannada_digits_to_english(clean_text(cell.get("text", "")))

        if "ಸರ್ವೆ" in cell_text or "survey" in cell_text.lower():
            survey_from_cells = (
                find_first(r"ಸರ್ವೆ\s*ಸಂಬರು\s*([A-Za-z0-9\/\-]+)", cell_text)
                or find_first(r"ಸರ್ವೆ\s*ನಂ\.?\s*([A-Za-z0-9\/\-]+)", cell_text)
                or find_first(r"\b(\d{1,4})\b", cell_text)
            )

            if survey_from_cells:
                break

    return (
        survey_from_cells
        or find_first(r"ಸರ್ವೆ\s*ಸಂಬರು\s*([A-Za-z0-9\/\-]+)", combined)
        or find_first(r"ಸರ್ವೆ\s*ನಂ\.?\s*([A-Za-z0-9\/\-]+)", combined)
        or fallback.get("1_survey_number", {}).get("survey_number")
    )


def extract_hissa(
    cells: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Optional[str]:

    combined = kannada_digits_to_english(text)

    hissa_from_cells = None

    for cell in cells:
        cell_text = kannada_digits_to_english(clean_text(cell.get("text", "")))

        if "ಹಿಸ್ಸಾ" in cell_text or "ಹಿಸಾ" in cell_text:
            hissa_from_cells = (
                find_first(r"ಹಿಸ್ಸಾ\s*([A-Za-z0-9\/\-]+)", cell_text)
                or find_first(r"ಹಿಸಾ\s*([A-Za-z0-9\/\-]+)", cell_text)
            )

            if hissa_from_cells:
                break

    return (
        hissa_from_cells
        or find_first(r"ಹಿಸ್ಸಾ\s*([A-Za-z0-9\/\-]+)", combined)
        or find_first(r"ಹಿಸಾ\s*([A-Za-z0-9\/\-]+)", combined)
        or fallback.get("2_hissa", {}).get("hissa_number")
    )


def extract_khata_vara(
    cells: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:

    old = fallback.get("3_khata_vara", {}) if fallback else {}

    combined = text

    # Top-left/central table area usually contains extent/kharab values.
    main_text = build_raw_region_text(
        cells,
        min_y_ratio=0.10,
        max_y_ratio=0.35,
        min_x_ratio=0.05,
        max_x_ratio=0.50,
    )

    extents = find_extents(main_text) or find_extents(combined)

    total_extent = old.get("total_extent")
    phut_kharab_a = old.get("phut_kharab_a")
    phut_kharab_b = old.get("phut_kharab_b") or "0.00"
    remaining_extent = old.get("remaining_extent")

    if len(extents) >= 1:
        total_extent = extents[0]

    if len(extents) >= 2:
        phut_kharab_a = extents[1]

    if len(extents) >= 3:
        remaining_extent = extents[-1]

    return {
        "total_extent": total_extent,
        "phut_kharab_a": phut_kharab_a,
        "phut_kharab_b": phut_kharab_b,
        "remaining_extent": remaining_extent,
        "raw_text": main_text or None,
    }


def extract_land_revenue(
    cells: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:

    old = fallback.get("4_land_revenue", {}) if fallback else {}

    revenue_region = build_raw_region_text(
        cells,
        min_y_ratio=0.10,
        max_y_ratio=0.35,
        min_x_ratio=0.25,
        max_x_ratio=0.55,
    )

    values = find_decimal_values(revenue_region)

    if len(values) >= 5:
        return {
            "land_revenue": values[0],
            "jodi": values[1],
            "cess": values[2],
            "water_rate": values[3],
            "total_revenue": values[4],
            "raw_text": revenue_region,
        }

    return {
        "land_revenue": old.get("land_revenue"),
        "jodi": old.get("jodi"),
        "cess": old.get("cess"),
        "water_rate": old.get("water_rate"),
        "total_revenue": old.get("total_revenue"),
        "raw_text": revenue_region or None,
    }


def extract_soil_type(
    cells: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:

    main_text = build_raw_region_text(
        cells,
        min_y_ratio=0.20,
        max_y_ratio=0.55,
        min_x_ratio=0.05,
        max_x_ratio=0.45,
    )

    combined = f"{main_text}\n{text}"

    for keyword in ["ಕೆಂಪು", "ಕಪ್ಪು", "ಕೆಬ್ಬೆ", "ಮರಳು", "red", "black"]:
        if keyword.lower() in combined.lower():
            return {
                "soil_type": keyword,
                "extraction_status": "extracted",
                "raw_text": main_text or None,
            }

    return {
        "soil_type": None,
        "extraction_status": "requires_layout_or_manual_review",
        "raw_text": main_text or None,
    }


def extract_land_classification(
    cells: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:

    main_text = build_raw_region_text(
        cells,
        min_y_ratio=0.20,
        max_y_ratio=0.55,
        min_x_ratio=0.05,
        max_x_ratio=0.45,
    )

    combined = f"{main_text}\n{text}"

    if "ಸರ್ಕಾರಿ" in combined:
        return {
            "land_classification": "ಸರ್ಕಾರಿ",
            "extraction_status": "extracted",
            "raw_text": main_text or None,
        }

    if "ಪಟ್ಟಾ" in combined:
        return {
            "land_classification": "ಪಟ್ಟಾ",
            "extraction_status": "extracted",
            "raw_text": main_text or None,
        }

    return {
        "land_classification": None,
        "extraction_status": "requires_layout_or_manual_review",
        "raw_text": main_text or None,
    }


def extract_tree_details(
    cells: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:

    main_text = build_raw_region_text(
        cells,
        min_y_ratio=0.20,
        max_y_ratio=0.55,
        min_x_ratio=0.05,
        max_x_ratio=0.45,
    )

    combined = kannada_digits_to_english(f"{main_text}\n{text}")

    tree_count = find_first(
        r"ಮರಗಳ\s*ಸಂಖ್ಯೆ.*?(\d+)",
        combined
    )

    return {
        "tree_count": tree_count,
        "tree_types": [],
        "extraction_status": (
            "extracted"
            if tree_count
            else "requires_layout_or_manual_review"
        ),
        "raw_text": main_text or None,
    }


# ---------------------------------------------------------
# BLOCK 8 - IRRIGATION
# ---------------------------------------------------------

def extract_irrigation_details(
    cells: List[Dict[str, Any]]
) -> Dict[str, Any]:

    irrigation_text = build_raw_region_text(
        cells,
        min_y_ratio=0.25,
        max_y_ratio=0.55,
        min_x_ratio=0.10,
        max_x_ratio=0.55,
    )

    values = extract_first_decimal_list(irrigation_text)

    return {
        "source": None,
        "kharif_extent": values[0] if len(values) > 0 else None,
        "rabi_extent": values[1] if len(values) > 1 else None,
        "garden_extent": values[2] if len(values) > 2 else None,
        "total_extent": values[3] if len(values) > 3 else None,
        "raw_text": irrigation_text or None,
        "extraction_status": (
            "extracted_partial"
            if irrigation_text
            else "requires_layout_or_manual_review"
        ),
    }


# ---------------------------------------------------------
# BLOCK 9 - OWNERSHIP / OCCUPANT
# ---------------------------------------------------------

def extract_owner_candidate_text(
    cells: List[Dict[str, Any]]
) -> str:

    # Owner/occupant usually lies in the right-middle main RTC table.
    # This is coordinate-region based, not keyword based.
    owner_region = build_raw_region_text(
        cells,
        min_y_ratio=0.12,
        max_y_ratio=0.55,
        min_x_ratio=0.45,
        max_x_ratio=0.78,
    )

    candidates = []

    for line in owner_region.splitlines():
        line = clean_text(line)

        if not line:
            continue

        if is_noise_text(line):
            continue

        # avoid pure revenue/extent lines
        if re.fullmatch(r"[\d\.\s\/\-]+", kannada_digits_to_english(line)):
            continue

        candidates.append(line)

    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    return ""


def extract_owner_details(
    cells: List[Dict[str, Any]],
    fallback: Dict[str, Any]
) -> List[Dict[str, Any]]:

    owner_text = extract_owner_candidate_text(cells)

    if not owner_text:
        # Fallback to OCR-based extraction from old extractor if present.
        old_owners = fallback.get("9_ownership_details", [])

        if old_owners:
            return old_owners

    if not owner_text:
        return []

    extent_values = find_extents(all_text(cells))

    return [
        {
            "owner_name": owner_text,
            "relation_type": None,
            "father_or_spouse_name": None,
            "address": None,
            "extent": extent_values[0] if extent_values else None,
            "khata_no": None,
            "possession_type": None,
            "raw_text": owner_text,
            "source": "layout_region_owner_block",
        }
    ]


def extract_possession_details(
    owners: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:

    return {
        "possession_type": (
            owners[0].get("possession_type")
            if owners
            else None
        ),
        "owner_wise_possession": [
            {
                "owner_name": owner.get("owner_name"),
                "possession_type": owner.get("possession_type"),
            }
            for owner in owners
        ],
    }


# ---------------------------------------------------------
# BLOCK 11 - RIGHTS / LIABILITIES
# ---------------------------------------------------------

def find_mr_number(text: str) -> Optional[str]:
    text = kannada_digits_to_english(text)

    return (
        find_first(r"(MR\s*[A-Z]?\d+\/\d{4}-\d{4})", text)
        or find_first(r"(MR\s*[A-Z0-9]+\/\d{4}-\d{4})", text)
    )


def find_court_reference(text: str) -> Optional[str]:
    text = kannada_digits_to_english(text)

    return (
        find_first(r"(noRA\/BET\/\d+\/\d+)", text)
        or find_first(r"(NORA\/BET\/\d+\/\d+)", text)
        or find_first(r"(RA\/BET\/\d+\/\d+)", text)
        or find_first(r"(RRT\/[A-Z0-9\/.\-]+)", text)
        or find_first(r"(NA\.CR\.[0-9\/\-]+)", text)
    )


def extract_rights_liabilities(
    text: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:

    old = fallback.get("11_rights_liabilities", {}) if fallback else {}

    mr_number = find_mr_number(text) or old.get("mr_number")
    court_reference = find_court_reference(text) or old.get("court_reference")

    recorded_on = (
        find_first(r"(\d{2}\/\d{2}\/\d{4})", text)
        or old.get("recorded_on")
    )

    court_stay = (
        True
        if "court stay" in text.lower()
        else old.get("court_stay", False)
    )

    return {
        "mr_number": mr_number,
        "court_stay": court_stay,
        "court_reference": court_reference,
        "recorded_on": recorded_on,
        "rights": old.get("rights"),
        "liabilities": old.get("liabilities"),
        "remarks": old.get("remarks"),
        "raw_text": text or None,
    }


# ---------------------------------------------------------
# BLOCK 12 - CULTIVATION
# ---------------------------------------------------------

def extract_years_and_seasons(
    text: str
) -> List[Dict[str, Optional[str]]]:

    text = kannada_digits_to_english(text)

    # First capture year and nearby text.
    year_matches = list(
        re.finditer(
            r"(20\d{2}-20\d{2})",
            text,
            flags=re.UNICODE
        )
    )

    result = []

    for match in year_matches:
        year = match.group(1)

        window = text[
            match.start():
            match.start() + 250
        ]

        season = None

        if "ಮುಂಗಾರು" in window:
            season = "ಮುಂಗಾರು"
        elif "ಹಿಂಗಾರು" in window:
            season = "ಹಿಂಗಾರು"
        elif "ಬೇಸಿಗೆ" in window:
            season = "ಬೇಸಿಗೆ"

        result.append({
            "crop_year": year,
            "season": season,
            "raw_window": clean_text(window),
        })

    return result


def extract_cultivator_name_from_window(
    window: str
) -> Optional[str]:

    window = clean_text(window)

    if not window:
        return None

    # Remove year.
    window = re.sub(
        r"20\d{2}-20\d{2}",
        " ",
        kannada_digits_to_english(window)
    )

    # Remove extents and decimals.
    window = re.sub(
        r"\d+\.\d+\.\d+\.\d+|\d+\.\d+",
        " ",
        window
    )

    # Remove known non-name words.
    remove_words = [
        "ಮುಂಗಾರು",
        "ಹಿಂಗಾರು",
        "ಬೇಸಿಗೆ",
        "ಅಮಿಶ್ರ",
        "ಮಿಶ್ರ",
        "ಒಟ್ಟು",
        "ವಿಸ್ತೀರ್ಣ",
        "ಎ",
        "ಗುಂ",
    ]

    for word in remove_words:
        window = window.replace(word, " ")

    window = clean_text(window)

    if not window:
        return None

    # Take first remaining phrase.
    parts = [
        clean_text(p)
        for p in re.split(r"\s{2,}|,", window)
        if clean_text(p)
    ]

    return parts[0] if parts else window

print("AGENTS RTC MAPPER RUNNING")
def extract_cultivation_details(  
    cells: List[Dict[str, Any]],
    text: str
) -> List[Dict[str, Any]]:

    cultivation_text = build_raw_region_text(
        cells,
        min_y_ratio=0.48,
        max_y_ratio=0.90,
        min_x_ratio=0.00,
        max_x_ratio=1.00,
    )

    combined = f"{cultivation_text}\n{text}"

    year_items = extract_years_and_seasons(combined)

    if not year_items:
        return []

    records = []

    extent_values = extract_first_decimal_list(cultivation_text)

    for index, item in enumerate(year_items):
        cultivator_name = extract_cultivator_name_from_window(
            item.get("raw_window", "")
        )

        records.append({
            "crop_year": item.get("crop_year"),
            "season": item.get("season"),
            "cultivator_name": cultivator_name,
            "cultivation_method": None,
            "land_use": None,
            "crop_name": None,
            "crop_extent": None,
            "irrigated_extent": None,
            "wet_extent": extent_values[index] if index < len(extent_values) else None,
            "dry_extent": extent_values[index + 1] if index + 1 < len(extent_values) else None,
            "total_crop_extent": None,
            "irrigation_source": None,
            "liability_amount": None,
            "mixed_crop": None,
            "mixed_crop_extent": None,
            "raw_text": item.get("raw_window"),
            "extraction_status": "extracted_partial",
        })

    return records


# ---------------------------------------------------------
# BLOCK 13 - OTHER RIGHTS
# ---------------------------------------------------------

def extract_other_rights(
    text: str,
    fallback: Dict[str, Any],
    cells: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:

    old = fallback.get("13_other_rights", {}) if fallback else {}

    raw_text = None

    if cells:
        raw_text = build_raw_region_text(
            cells,
            min_y_ratio=0.82,
            max_y_ratio=1.00,
            min_x_ratio=0.00,
            max_x_ratio=1.00,
        )

    combined = f"{raw_text or ''}\n{text}"

    restrictions = None

    if "court stay" in combined.lower():
        restrictions = "court stay"

    return {
        "other_rights": old.get("other_rights"),
        "restrictions": restrictions or old.get("restrictions"),
        "remarks": old.get("remarks"),
        "raw_text": raw_text or None,
    }


# ---------------------------------------------------------
# MAIN MAPPER
# ---------------------------------------------------------

def map_rtc_cells_to_blocks(
    cells_with_text: List[Dict[str, Any]],
    table_layout: Optional[Dict[str, Any]] = None,
    fallback_extraction: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    fallback_extraction = fallback_extraction or {}
    text = all_text(cells_with_text)

    owners = extract_owner_details(
        cells_with_text,
        fallback_extraction
    )

    return {
        "document_type": "RTC",

        "header": extract_header_from_fallback(
            fallback_extraction
        ),

        "1_survey_number": {
            "survey_number": extract_survey_number(
                cells_with_text,
                text,
                fallback_extraction
            )
        },

        "2_hissa": {
            "hissa_number": extract_hissa(
                cells_with_text,
                text,
                fallback_extraction
            )
        },

        "3_khata_vara": extract_khata_vara(
            cells_with_text,
            text,
            fallback_extraction
        ),

        "4_land_revenue": extract_land_revenue(
            cells_with_text,
            text,
            fallback_extraction
        ),

        "5_soil_type": extract_soil_type(
            cells_with_text,
            text
        ),

        "6_land_classification": extract_land_classification(
            cells_with_text,
            text
        ),

        "7_tree_details": extract_tree_details(
            cells_with_text,
            text
        ),

        "8_irrigation_details": extract_irrigation_details(
            cells_with_text
        ),

        "9_ownership_details": owners,

        "10_possession_details": extract_possession_details(
            owners,
            text
        ),

        "11_rights_liabilities": extract_rights_liabilities(
            text,
            fallback_extraction
        ),

        "12_cultivation_details": extract_cultivation_details(
            cells_with_text,
            text
        ),

        "13_other_rights": extract_other_rights(
            text,
            fallback_extraction,
            cells_with_text
        ),

        "raw_regions": {
            "block_8_irrigation_raw": build_raw_region_text(
                cells_with_text,
                min_y_ratio=0.25,
                max_y_ratio=0.55,
                min_x_ratio=0.10,
                max_x_ratio=0.55,
            ),
            "block_9_ownership_raw": build_raw_region_text(
                cells_with_text,
                min_y_ratio=0.12,
                max_y_ratio=0.55,
                min_x_ratio=0.45,
                max_x_ratio=0.78,
            ),
            "block_12_cultivation_raw": build_raw_region_text(
                cells_with_text,
                min_y_ratio=0.48,
                max_y_ratio=0.90,
                min_x_ratio=0.00,
                max_x_ratio=1.00,
            ),
            "block_13_other_rights_raw": build_raw_region_text(
                cells_with_text,
                min_y_ratio=0.82,
                max_y_ratio=1.00,
                min_x_ratio=0.00,
                max_x_ratio=1.00,
            ),
        },

        "extraction_source": "cell_ocr_layout_region_mapper_v3",
    }


def map_rtc_cells(
    cells_with_text,
    table_layout=None,
    fallback_extraction=None
):
    return map_rtc_cells_to_blocks(
        cells_with_text=cells_with_text,
        table_layout=table_layout,
        fallback_extraction=fallback_extraction,
    )
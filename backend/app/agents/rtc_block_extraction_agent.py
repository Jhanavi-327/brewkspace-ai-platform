import re
from typing import Any, Dict, Optional, List


def clean_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = re.sub(r"\s+", " ", value.strip())
    value = value.replace(":", "").replace("：", "").strip()

    return value


def find_first(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE | re.UNICODE | re.DOTALL)
    return clean_value(match.group(1)) if match else None


def find_all_extents(text: str) -> List[str]:
    return re.findall(r"\d+\.\d+\.\d+\.\d+", text)


def get_ocr_section(text: str, section_name: str) -> str:
    marker = f"--- {section_name} OCR ---"

    if marker not in text:
        return text

    part = text.split(marker, 1)[1]

    if section_name == "TESSERACT" and "--- EASYOCR OCR ---" in part:
        part = part.split("--- EASYOCR OCR ---", 1)[0]

    return part


def extract_header(text: str) -> Dict[str, Any]:
    header = {
        "taluk": None,
        "hobli": None,
        "village": None,
        "valid_from": None,
        "valid_to": None,
    }

    lines = [
        clean_value(line)
        for line in text.splitlines()
        if clean_value(line)
    ]

    for i, line in enumerate(lines):

        # Taluk, hobli and village are printed across ONE header row, and OCR
        # often returns that row as a single line:
        #
        #   ತಾಲ್ಲೂಕು ಮೊಹರು ತಾಲ್ಲೂಕು: ಯಲಹ೦ಕ ಹೋಬಳಿ: ಜಾಲ2 ಗ್ರಾಮ: ಕುದುರೆಗೆರೆ ಪುಟದ ...
        #
        # Capturing "everything after the label" then hands the whole remainder
        # back as the value — a screenshot of the Yalahanka RTC reported its
        # taluk as "ತಾಲ್ಲೂಕು ಯಲಹ೦ಕ ಹೋಬಳಿ . ಹಾಲ2 ಗ್ರಾಮ *ುದುರೆಗೆರೆ ಪುಟದ ಕ್ರಮ
        # ಸಂಖ್ಯೆ". Each value has to stop where the next label begins.
        def cut_at(value, stops):
            for stop in stops:
                value = value.split(stop)[0]

            # clean_value returns None for an empty string.
            return (clean_value(value) or "").strip(" :：.,-")

        # Taluk
        if "ತಾಲ್ಲೂಕು" in line or "ತಾಲ್ಲೂ ಕು" in line:
            # Greedy prefix: the row prints "ತಾಲ್ಲೂಕು ಮೊಹರು" (the taluk SEAL)
            # before the taluk itself, so the last label on the line is the
            # one carrying the value.
            value = re.sub(r".*ತಾಲ್ಲೂ\s*ಕು\s*[:：]?\s*", "", line)
            value = value.replace("ಮೊಹರು", "")
            value = cut_at(value, ("ಹೋಬಳಿ", "ಹೋಬಲಿ", "ಹೋಬ", "ಗ್ರಾಮ", "ಪುಟ"))

            if value:
                header["taluk"] = value

        # Hobli
        if "ಹೋಬಳಿ" in line or "ಹೋಬಲಿ" in line:
            match = re.search(r"ಹೋಬ[ಳಲ]ಿ\s*[:：]?\s*(.*)", line)
            if match:
                value = cut_at(match.group(1), ("ಗ್ರಾಮ", "ಪುಟ"))

                if value:
                    header["hobli"] = value
                elif i + 1 < len(lines):
                    header["hobli"] = clean_value(lines[i + 1])

        # Village
        if "ಗ್ರಾಮ" in line:
            match = re.search(r"ಗ್ರಾಮ\s*[:：]?\s*(.*)", line)
            if match:
                value = cut_at(match.group(1), ("ಪುಟ",))

                # If OCR puts village on next line
                if not value and i + 1 < len(lines):
                    value = clean_value(lines[i + 1])

                # "ಗ್ರಾಮ ನಮೂನೆ 2" (Village Form No. 2) is printed in the corner
                # of every RTC and matches the village label first; OCR drops
                # the trailing vowel often enough to need both spellings.
                if value and not any(
                    bad in value
                    for bad in ("ಪುಟ", "ನಮೂನ", "ನಮೂನೆ", "Valid", "ಫಾರಂ")
                ):
                    header["village"] = value

    # Bhoomi prints the validity window in TWO formats, and only one was read:
    #
    #   Valid from 14/09/2020 To Till Date                    (Vajarahalli)
    #   Valid from 2010-05-28 16:47:00 To 2021-02-05 10:41:00 (RTC 59-P4)
    #
    # The second is the older ISO-with-timestamp form, and matching only
    # DD/MM/YYYY lost BOTH dates on every RTC that uses it. The ISO readings are
    # normalised to DD/MM/YYYY so the field means the same thing whichever
    # format the source document happened to print.
    def normalise_date(value):
        if not value:
            return None

        iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)

        if iso:
            return f"{iso.group(3)}/{iso.group(2)}/{iso.group(1)}"

        return value

    header["valid_from"] = normalise_date(
        find_first(r"Valid from\s*([0-9]{2}\/[0-9]{2}\/[0-9]{4})", text)
        or find_first(r"Valid from\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
        or find_first(r"from\s*([0-9]{2}\/[0-9]{2}\/[0-9]{4})", text)
        or find_first(r"from\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    )

    header["valid_to"] = normalise_date(
        find_first(r"To\s*(Till Date)", text)
        or find_first(r"To\s*([0-9]{2}\/[0-9]{2}\/[0-9]{4})", text)
        or find_first(r"To\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    )

    return header

def extract_1_survey_number(text: str) -> Dict[str, Any]:
    survey_number = (
        find_first(r"\n\s*(\d{1,4})\s*\n\s*ಪೂಟ್", text)
        or find_first(r"\n\s*(\d{1,4})\s+ಪೂಟ್", text)
        or find_first(r"ಸರ್ವೆ\s*\n?\s*(\d{1,4})", text)
        or find_first(r"ಸರ್ವೆ\s*ನಂ\.?\s*(\d{1,4})", text)
    )

    return {
        "survey_number": survey_number
    }


def extract_2_hissa(text: str) -> Dict[str, Any]:
    hissa = (
        find_first(r"2\.\s*\n.*?\n\s*(\d{1,3})\s*\n\s*ಒಟ್ಟು", text)
        or find_first(r"2\.\s*\n\s*(\d{1,3})\s*\n\s*ಒಟ್ಟು", text)
        or find_first(r"ಹಿಸ್ಸಾ\s*\n?\s*(\d{1,3})", text)
        or find_first(r"ಹಿಸಾ\s*\n?\s*(\d{1,3})", text)
        or find_first(r"\b(P\d+)\b", text)
    )

    return {
        "hissa_number": hissa
    }


def extract_3_khata_vara(text: str) -> Dict[str, Any]:
    extents = find_all_extents(text)
    non_zero_extents = [x for x in extents if not x.startswith("0.00")]

    total_extent = non_zero_extents[0] if non_zero_extents else None

    phut_kharab_a = find_first(
        r"ಪೂಟ್\s*ಖರಾಬ್\s*\(ಅ\).*?(\d+\.\d+\.\d+\.\d+)",
        text
    )

    phut_kharab_b = find_first(
        r"ಪೂಟ್\s*ಖರಾಬ್\s*\(\s*ಬ\s*\).*?(\d+\.\d+\.\d+\.\d+|\d+\.\d+)",
        text
    )

    remaining_extent = find_first(
        r"ಉಳಿದದ್ದು\s*(\d+\.\d+\.\d+\.\d+)",
        text
    )

    return {
        "total_extent": total_extent,
        "phut_kharab_a": phut_kharab_a,
        "phut_kharab_b": phut_kharab_b,
        "remaining_extent": remaining_extent
    }


def extract_4_land_revenue(text: str) -> Dict[str, Any]:
    land_revenue = None

    # In RTC, land revenue usually appears after total extent in the same top table.
    # Example:
    # 2.26.00.00
    # 5.46
    revenue_after_extent = re.search(
        r"\d+\.\d+\.\d+\.\d+\s*\n\s*(\d+\.\d{2})",
        text,
        re.IGNORECASE | re.UNICODE
    )

    if revenue_after_extent:
        land_revenue = clean_value(revenue_after_extent.group(1))

    if not land_revenue:
        land_revenue = (
            find_first(r"ಭೂ\s*ಕಂದಾಯ\s*\n?\s*(\d+\.\d+)", text)
            or find_first(r"ಕಂದಾಯ.*?\n\s*(\d+\.\d+)", text)
        )

    jodi = find_first(r"ಜೋಡಿ\s*\n?\s*(\d+\.\d+)", text)
    cess = find_first(r"ಸೆಸ್ಸುಗಳು\s*\n?\s*(\d+\.\d+)", text)
    water_rate = find_first(r"ನೀರಿನ\s*ದರ\s*\n?\s*(\d+\.\d+)", text)

    total_revenue = (
        find_first(r"ಒಟ್ಟು\s*\n?\s*(\d+\.\d+)", text)
        or land_revenue
    )

    return {
        "land_revenue": land_revenue,
        "jodi": jodi,
        "cess": cess,
        "water_rate": water_rate,
        "total_revenue": total_revenue
    }


def extract_5_soil_type(text: str) -> Dict[str, Any]:
    soil_type = (
        find_first(r"5\.\s*ಮಣ್ಣ.*?\n([^\n]+)", text)
        or find_first(r"\n(ಕೆಂಪು|ಕಪ್ಪು|ಕೆಬ್ಬೆ|ಮರಳು)\n", text)
    )

    return {
        "soil_type": soil_type
    }


def extract_6_land_classification(text: str) -> Dict[str, Any]:
    classification = (
        find_first(r"6\.\s*ಪಟ್ಟಾ\s*\n?\s*([^\n]+)", text)
        or find_first(r"\n(ಸರ್ಕಾರಿ|ಪಟ್ಟಾ)\n", text)
    )

    return {
        "land_classification": classification
    }


def extract_7_tree_details(text: str) -> Dict[str, Any]:
    tree_count = find_first(
        r"7\.\s*ಮರಗಳ\s*ಸಂಖ್ಯ.*?\n\s*(\d+)",
        text
    )

    return {
        "tree_count": tree_count,
        "tree_types": []
    }


def extract_8_irrigation_details(text: str) -> Dict[str, Any]:
    return {
        "source": None,
        "kharif_extent": None,
        "rabi_extent": None,
        "garden_extent": None,
        "total_extent": None
    }


def extract_9_ownership_details(text: str) -> List[Dict[str, Any]]:
    stop_markers = [
        "L2. ಸಾಗುವ",
        "12. ಸಾಗುವ",
        "ವರ್ಷ ವ್ಯವಸಾಯಗಾರನ",
    ]

    section = text

    for marker in stop_markers:
        if marker in section:
            section = section.split(marker)[0]

    lines = [clean_value(line) for line in section.splitlines()]
    lines = [line for line in lines if line]

    skip_keywords = [
        "ತಾಲ್ಲೂ", "ಹೋಬಳಿ", "ಗ್ರಾಮ", "ಸರ್ವೆ", "ಪೂಟ್", "ಖರಾಬ್",
        "ಕಂದಾಯ", "ಒಟ್ಟು", "ಉಳಿದದ್ದು", "ನೀರಿನ", "MR", "court",
        "order", "Valid", "Print", "First", "Previous", "Next", "Last",
        "ಖಾತೆ", "ಹಕ್ಕು", "ಋಣ", "ವರ್ಷ", "ಮುಂಗಾರು", "ಹಿಂಗಾರು",
        "ಬಾಗಾಯ್ತು", "ಬೆಳೆ", "ವಿಸ್ತೀರ್ಣ", "ನಮೂನ", "ನಂ", "ರೂ",
        "ಎಕರೆ", "ಗುಂಟೆ", "ಕಬ್ಜ", "ಧೀನ", "ಮೂಲ", "ಸಂಖ್ಯ", "0.00",
        "2025", "2026"
    ]

    relation_keywords = ["ಬಿನ್", "ಕೋಂ", "ಲೇ"]

    owners = []

    for i, line in enumerate(lines):
        if any(skip.lower() in line.lower() for skip in skip_keywords):
            continue

        if any(rel in line for rel in relation_keywords):
            owner_name = line

            if i + 1 < len(lines):
                next_line = lines[i + 1]

                if (
                    not any(skip.lower() in next_line.lower() for skip in skip_keywords)
                    and 2 <= len(next_line) <= 40
                    and not any(rel in next_line for rel in relation_keywords)
                ):
                    owner_name = f"{owner_name} {next_line}"

            owner_name = owner_name.replace("ಮೇಲಿನ ಜಂಟಿ", "").strip()
            owner_name = owner_name.replace("ಪೇಲಿನ ಜ", "").strip()
            owner_name = owner_name.replace("ಮೇಲಿನ ಜ", "").strip()

            if owner_name and owner_name not in [o["owner_name"] for o in owners]:
                owners.append({
                    "owner_name": owner_name,
                    "father_or_spouse_name": None,
                    "address": None,
                    "extent": None,
                    "khata_no": None,
                    "possession_type": None
                })

    return owners


def extract_10_possession_details(text: str) -> Dict[str, Any]:
    possession_types = re.findall(
        r"(ಮೇಲಿನ\s*ಜಂಟಿ|ಸ್ವಂತ|ಪೌತಿ|ಖರೀದಿ|ವಾರಸುದಾರ)",
        text,
        re.IGNORECASE | re.UNICODE
    )

    return {
        "possession_type": list(set(possession_types)) if possession_types else None
    }


def extract_11_rights_liabilities(text: str) -> Dict[str, Any]:
    mr_number = (
        find_first(r"(MR\s*[A-Z]?\d+\/\d{4}-\d{4})", text)
        or find_first(r"(MR\s*[A-Z0-9]+\/\d{4}-\d{4})", text)
    )

    court_stay = "court stay" in text.lower()

    court_reference = (
        find_first(r"(noRA\/BET\/\d+\/\d+)", text)
        or find_first(r"(RRT\/[A-Z0-9\/.\-]+)", text)
        or find_first(r"(NA\.CR\.[0-9\/\-]+)", text)
    )

    recorded_on = find_first(
        r"(\d{2}\/\d{2}\/\d{4})\s*\n\s*order",
        text
    )

    return {
        "mr_number": mr_number,
        "court_stay": court_stay,
        "court_reference": court_reference,
        "recorded_on": recorded_on,
        "rights": None,
        "liabilities": None,
        "remarks": None
    }


def extract_12_cultivation_details(text: str) -> List[Dict[str, Any]]:
    crop_rows = []

    crop_years = re.findall(r"(20\d{2}-20\d{2})", text)

    for year in sorted(set(crop_years)):
        crop_rows.append({
            "crop_year": year,
            "season": None,
            "cultivator_name": None,
            "cultivation_method": None,
            "land_use": None,
            "crop_name": None,
            "crop_extent": None,
            "irrigated_extent": None
        })

    return crop_rows


def extract_13_other_rights(text: str) -> Dict[str, Any]:
    restrictions = None

    if "court stay" in text.lower():
        restrictions = "court stay"

    return {
        "other_rights": None,
        "restrictions": restrictions,
        "remarks": None
    }


def extract_rtc_blocks(text: str) -> Dict[str, Any]:
    tesseract_text = get_ocr_section(text, "TESSERACT")
    easyocr_text = get_ocr_section(text, "EASYOCR")

    main_text = easyocr_text if len(easyocr_text) > len(tesseract_text) else text

    return {
        "document_type": "RTC",

        "header": extract_header(tesseract_text) or extract_header(main_text),

        "1_survey_number": extract_1_survey_number(main_text),

        "2_hissa": extract_2_hissa(main_text),

        "3_khata_vara": extract_3_khata_vara(main_text),

        "4_land_revenue": extract_4_land_revenue(main_text),

        "5_soil_type": extract_5_soil_type(main_text),

        "6_land_classification": extract_6_land_classification(main_text),

        "7_tree_details": extract_7_tree_details(main_text),

        "8_irrigation_details": extract_8_irrigation_details(main_text),

        "9_ownership_details": extract_9_ownership_details(main_text),

        "10_possession_details": extract_10_possession_details(main_text),

        "11_rights_liabilities": extract_11_rights_liabilities(main_text),

        "12_cultivation_details": extract_12_cultivation_details(main_text),

        "13_other_rights": extract_13_other_rights(main_text),
    }
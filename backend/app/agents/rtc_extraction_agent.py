import re
from typing import Dict, Any, Optional

from app.services.location_normalizer import normalize_location_fields


def find_first(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE | re.UNICODE | re.DOTALL)
    return match.group(1).strip() if match else None


def clean_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value.strip())
    value = value.replace(":", "").replace("：", "").strip()
    return value


def clean_taluk(value: Optional[str]) -> Optional[str]:
    value = clean_value(value)
    if not value:
        return None

    value = value.replace("ಮೊಹರು", "")
    value = value.replace("ತಾಲ್ಲೂಕು", "")
    value = value.replace("ತಾಲ್ಲೂ ಕು", "")
    return clean_value(value)


def clean_hobli(value: Optional[str]) -> Optional[str]:
    value = clean_value(value)
    if not value:
        return None

    value = value.lstrip(".").strip()
    value = value.replace("ಶೆ.", "ಕೆ.")
    value = value.replace("ಆರ್‌", "ಆರ್")
    value = value.replace("ಆರ್.", "ಆರ್")
    return clean_value(value)


def get_ocr_section(text: str, section_name: str) -> str:
    marker = f"--- {section_name} OCR ---"
    if marker not in text:
        return text

    part = text.split(marker, 1)[1]

    if section_name == "TESSERACT" and "--- EASYOCR OCR ---" in part:
        part = part.split("--- EASYOCR OCR ---", 1)[0]

    return part


def extract_header_location(text: str) -> Dict[str, Optional[str]]:
    patterns = [
        r"ತಾಲ್ಲೂಕು\s*ಮೊಹರು\s*ತಾಲ್ಲೂಕು\s*[:：]?\s*(.*?)\s+ಹೋಬಳಿ\s*[:：]?\s*(.*?)\s+ಗ್ರಾಮ\s*[:：]?\s*(.*?)\s+ಪುಟ",
        r"ತಾಲ್ಲೂ\s*ಕು\s*[:：]?\s*(.*?)\s+ಹೋಬಳಿ\s*[:：]?\s*(.*?)\s+ಗ್ರಾಮ\s*[:：]?\s*(.*?)\s+ಪುಟ",
        r"ತಾಲ್ಲೂ\s*ಕು\s*[:：]?\s*(.*?)\s+ಹೋಬಲಿ\s*[:：]?\s*(.*?)\s+ಗ್ರಾಮ\s*[:：]?\s*(.*?)\s+ಪುಟ",
        r"ತಾಲ್ಲೂಕು\s*[:：]?\s*(.*?)\s+ಹೋಬಳಿ\s*[:：]?\s*(.*?)\s+ಗ್ರಾಮ\s*[:：]?\s*(.*?)\s+ಪುಟ",
        r"ತಾಲ್ಲೂಕು\s*[:：]?\s*(.*?)\s+ಹೋಬಲಿ\s*[:：]?\s*(.*?)\s+ಗ್ರಾಮ\s*[:：]?\s*(.*?)\s+ಪುಟ",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.UNICODE | re.DOTALL)
        if match:
            return {
                "taluk": clean_taluk(match.group(1)),
                "hobli": clean_hobli(match.group(2)),
                "village": clean_value(match.group(3)),
            }

    return {"taluk": None, "hobli": None, "village": None}


def extract_survey_number(text: str) -> Optional[str]:
    patterns = [
        r"\n\s*(\d{1,4})\s*\n\s*ಪೂಟ್",
        r"\n\s*(\d{1,4})\s+ಪೂಟ್",
        r"ಸರ್ವೆ\s*\n?\s*(\d{1,4})",
        r"ಸರ್ವೆ\s*ನಂ\.?\s*(\d{1,4})",
        r"ಸರ್ವ\s*\n?\s*(\d{1,4})",
    ]

    for pattern in patterns:
        value = find_first(pattern, text)
        if value:
            return clean_value(value)

    return None


def extract_hissa(text: str) -> Optional[str]:
    patterns = [
        r"order\s*\n.*?\n\s*2\.\s*\n.*?\n\s*(\d{1,3})\s*\n\s*ಒಟ್ಟು",
        r"2\.\s*\n.*?\n\s*(\d{1,3})\s*\n\s*ಒಟ್ಟು",
        r"2\.\s*\n\s*(\d{1,3})\s*\n\s*ಒಟ್ಟು",
        r"ಹಿಸ್ಸಾ\s*\n?\s*(\d{1,3})",
        r"ಹಿಸಾ\s*\n?\s*(\d{1,3})",
        r"\b(P\d+)\b",
    ]

    for pattern in patterns:
        value = find_first(pattern, text)
        if value:
            return clean_value(value)

    return None


def extract_extent(text: str) -> Optional[str]:
    matches = re.findall(r"\d+\.\d+\.\d+\.\d+", text)

    if not matches:
        return None

    non_zero = [
        value for value in matches
        if not value.startswith("0.00")
    ]

    return clean_value(non_zero[0] if non_zero else matches[0])


def extract_mr_number(text: str) -> Optional[str]:
    patterns = [
        r"(MR\s*[A-Z]?\d+\/\d{4}-\d{4})",
        r"(MR\s*[A-Z0-9]+\/\d{4}-\d{4})",
        r"(MR\s*[A-Z]?\d+\s*\/\s*\d{4}\s*-\s*\d{4})",
    ]

    for pattern in patterns:
        value = find_first(pattern, text)
        if value:
            return clean_value(value)

    return None


def extract_conversion_reference(text: str) -> Optional[str]:
    patterns = [
        r"(RRT[\/I1]?[0-9A-Z\/.\-]+)",
        r"(RRT\/[0-9A-Z\/.\-]+)",
        r"(NA\.CR\.[0-9\/\-]+)",
    ]

    for pattern in patterns:
        value = find_first(pattern, text)
        if value:
            return clean_value(value.replace("I", "/").replace("l", "/"))

    return None


def extract_valid_from(text: str) -> Optional[str]:
    patterns = [
        r"Valid from\s*([0-9]{2}\/[0-9]{2}\/[0-9]{4})",
        r"from\s*([0-9]{2}\/[0-9]{2}\/[0-9]{4})",
        r"Valid from\s*([0-9]{4}[-.][0-9]{2}[-.][0-9]{2}\s+[0-9:.]+)",
    ]

    for pattern in patterns:
        value = find_first(pattern, text)
        if value:
            return clean_value(value)

    return None


def extract_valid_to(text: str) -> Optional[str]:
    patterns = [
        r"To\s*(Till Date)",
        r"To\s*([0-9]{2}\/[0-9]{2}\/[0-9]{4})",
        r"To\s*([0-9]{4}[-.][0-9]{2}[-.][0-9]{2}\s+[0-9:.]+)",
    ]

    for pattern in patterns:
        value = find_first(pattern, text)
        if value:
            return clean_value(value)

    return None


def extract_owner_names(text: str) -> Optional[list]:
    stop_markers = [
        "L2. ಸಾಗುವ",
        "12. ಸಾಗುವ",
        "ವರ್ಷ ವ್ಯವಸಾಯಗಾರನ",
    ]

    owner_section = text

    for marker in stop_markers:
        if marker in owner_section:
            owner_section = owner_section.split(marker)[0]

    names = []
    relationship_keywords = ["ಬಿನ್", "ಕೋಂ", "ಲೇ"]

    skip_keywords = [
        "ತಾಲ್ಲೂ", "ಹೋಬಳಿ", "ಹೋಬಲಿ", "ಗ್ರಾಮ", "ಸರ್ವೆ", "ಪೂಟ್", "ಖರಾಬ್",
        "ಕಂದಾಯ", "ಒಟ್ಟು", "ಉಳಿದದ್ದು", "ನೀರಿನ", "MR", "court", "order",
        "Valid", "Print", "First", "Previous", "Next", "Last", "ಖಾತೆ",
        "ಹಕ್ಕು", "ಋಣ", "ವರ್ಷ", "ಮುಂಗಾರು", "ಹಿಂಗಾರು", "ಬಾಗಾಯ್ತು",
        "ಬೆಳೆ", "ವಿಸ್ತೀರ್ಣ", "ನಮೂನ", "ನಂ", "ರೂ", "ಎಕರೆ", "ಗುಂಟೆ",
        "ಕಬ್ಜ", "ಧೀನ", "ಮೂಲ", "ಸಂಖ್ಯ", "0.00", "2025", "2026",
    ]

    lines = [clean_value(line) for line in owner_section.splitlines()]
    lines = [line for line in lines if line]

    for i, line in enumerate(lines):
        if any(skip.lower() in line.lower() for skip in skip_keywords):
            continue

        if any(rel in line for rel in relationship_keywords):
            name = line

            if i + 1 < len(lines):
                next_line = lines[i + 1]

                if (
                    not any(skip.lower() in next_line.lower() for skip in skip_keywords)
                    and 2 <= len(next_line) <= 40
                    and not any(rel in next_line for rel in relationship_keywords)
                ):
                    name = f"{name} {next_line}"

            name = name.replace("ಮೇಲಿನ ಜಂಟಿ", "").strip()
            name = name.replace("ಪೇಲಿನ ಜ", "").strip()
            name = name.replace("ಮೇಲಿನ ಜ", "").strip()

            if 4 <= len(name) <= 120 and name not in names:
                names.append(name)

    return names if names else None


def extract_rtc_fields(text: str) -> Dict[str, Any]:
    tesseract_text = get_ocr_section(text, "TESSERACT")
    easyocr_text = get_ocr_section(text, "EASYOCR")

    location = extract_header_location(tesseract_text)

    if not location["taluk"] or not location["hobli"] or not location["village"]:
        location = extract_header_location(easyocr_text)

    if not location["taluk"] or not location["hobli"] or not location["village"]:
        location = extract_header_location(text)

    fields = {
        "document_type": "RTC",
        "taluk": location["taluk"],
        "hobli": location["hobli"],
        "village": location["village"],
        "survey_number": (
            extract_survey_number(tesseract_text)
            or extract_survey_number(easyocr_text)
            or extract_survey_number(text)
        ),
        "hissa": (
            extract_hissa(tesseract_text)
            or extract_hissa(easyocr_text)
            or extract_hissa(text)
        ),
        "extent": (
            extract_extent(easyocr_text)
            or extract_extent(tesseract_text)
            or extract_extent(text)
        ),
        "mr_number": (
            extract_mr_number(easyocr_text)
            or extract_mr_number(tesseract_text)
            or extract_mr_number(text)
        ),
        "conversion_reference": (
            extract_conversion_reference(tesseract_text)
            or extract_conversion_reference(easyocr_text)
            or extract_conversion_reference(text)
        ),
        "valid_from": (
            extract_valid_from(tesseract_text)
            or extract_valid_from(easyocr_text)
            or extract_valid_from(text)
        ),
        "valid_to": (
            extract_valid_to(tesseract_text)
            or extract_valid_to(easyocr_text)
            or extract_valid_to(text)
        ),
        "owner_names": (
            extract_owner_names(easyocr_text)
            or extract_owner_names(tesseract_text)
            or extract_owner_names(text)
        ),
    }

    fields = normalize_location_fields(fields)

    missing_fields = [
        key for key, value in fields.items()
        if key != "document_type" and value in [None, "", []]
    ]

    return {
        "document_type": "RTC",
        "fields": fields,
        "missing_fields": missing_fields,
    }
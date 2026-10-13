import re
from datetime import datetime
from typing import Any, Mapping

from app.schemas.aadhaar import AadhaarCard


def clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError("Expected extracted text to be a string or null.")

    return value.strip() or None


def extract_birth_date_or_year(value: Any) -> str | None:
    text = clean_optional_text(value)

    if text is None:
        return None

    if re.fullmatch(r"[0-9]{4}", text):
        return text

    for date_format in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            parsed_date = datetime.strptime(text, date_format)
            return parsed_date.strftime("%d/%m/%Y")
        except ValueError:
            continue

    return None


def map_aadhaar_fields(
    extracted: Mapping[str, Any],
) -> AadhaarCard:
    return AadhaarCard.model_validate({
        "name": clean_optional_text(extracted.get("name")),
        "date_of_birth_or_year": extract_birth_date_or_year(
            extracted.get("date_of_birth_or_year")
        ),
        "gender": clean_optional_text(extracted.get("gender")),
        "aadhaar_number": clean_optional_text(
            extracted.get("aadhaar_number")
        ),
        "address": clean_optional_text(extracted.get("address")),
    })
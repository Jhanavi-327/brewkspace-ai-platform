import re
from datetime import datetime
from typing import Any, Mapping

from app.schemas.pan import PanCard


def clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError("Expected a string or null.")

    return value.strip() or None


def normalize_pan_number(value: Any) -> str | None:
    text = clean_optional_text(value)

    if text is None:
        return None

    text = text.upper()

    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text):
        return None

    return text


def normalize_date_of_birth(value: Any) -> str | None:
    text = clean_optional_text(value)

    if text is None:
        return None

    for date_format in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(text, date_format)
            return parsed.strftime("%d/%m/%Y")
        except ValueError:
            continue

    return None


def map_pan_fields(extracted: Mapping[str, Any]) -> PanCard:
    return PanCard.model_validate({
        "pan_number": normalize_pan_number(
            extracted.get("pan_number")
        ),
        "name": clean_optional_text(extracted.get("name")),
        "father_name": clean_optional_text(
            extracted.get("father_name")
        ),
        "date_of_birth": normalize_date_of_birth(
            extracted.get("date_of_birth")
        ),
    })
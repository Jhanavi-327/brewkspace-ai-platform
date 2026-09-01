from difflib import SequenceMatcher
from typing import Optional


MASTER_TALUKS = [
    "ಬೆಂಗಳೂರು(ಪೂರ್ವ)",
    "ಯಲಹಂಕ",
    "ದೇವನಹಳ್ಳಿ",
    "ದೊಡ್ಡಬಳ್ಳಾಪುರ",
]

MASTER_HOBLIS = [
    "ಕೆ.ಆರ್.ಪುರ 1",
    "ಕೆ.ಆರ್.ಪುರ 2",
    "ಬಿದರಹಳ್ಳಿ 1",
    "ಜಾಲ 1",
    "ಜಾಲ 2",
]

MASTER_VILLAGES = [
    "ರಾಚೇನಹಳ್ಳಿ",
    "ಕಾಡುಗೋಡಿ",
    "ಕುದುರೆಗೆರೆ",
    "ಸಾಡುಗೋಡಿ",
]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def normalize_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = value.strip()
    value = value.replace(" ", "")
    value = value.replace(".", "")
    value = value.replace("‌", "")
    value = value.replace("೦", "0")

    return value


def fuzzy_match(value: Optional[str], master_values: list[str], threshold: float = 0.78):
    if not value:
        return {
            "raw": value,
            "normalized": None,
            "match_score": 0,
            "matched": False,
        }

    value_norm = normalize_text(value)

    best_match = None
    best_score = 0

    for master in master_values:
        score = similarity(value_norm, normalize_text(master))

        if score > best_score:
            best_score = score
            best_match = master

    return {
        "raw": value,
        "normalized": best_match if best_score >= threshold else value,
        "match_score": round(best_score, 2),
        "matched": best_score >= threshold,
    }


def normalize_location_fields(fields: dict) -> dict:
    taluk_result = fuzzy_match(fields.get("taluk"), MASTER_TALUKS)
    hobli_result = fuzzy_match(fields.get("hobli"), MASTER_HOBLIS)
    village_result = fuzzy_match(fields.get("village"), MASTER_VILLAGES)

    fields["taluk_raw"] = fields.get("taluk")
    fields["hobli_raw"] = fields.get("hobli")
    fields["village_raw"] = fields.get("village")

    fields["taluk"] = taluk_result["normalized"]
    fields["hobli"] = hobli_result["normalized"]
    fields["village"] = village_result["normalized"]

    fields["location_validation"] = {
        "taluk": taluk_result,
        "hobli": hobli_result,
        "village": village_result,
    }

    return fields
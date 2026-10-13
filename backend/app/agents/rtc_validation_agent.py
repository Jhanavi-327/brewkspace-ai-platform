from typing import Dict, Any


def validate_rtc_extraction(
    ocr_text: str,
    extracted_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    MVP validator.
    Later replace this with Gemini/GPT validation.
    """

    fields = extracted_data.get("fields", {})

    warnings = []

    if not fields.get("taluk"):
        warnings.append("Taluk not extracted")

    if not fields.get("hobli"):
        warnings.append("Hobli not extracted")

    if not fields.get("village"):
        warnings.append("Village not extracted")

    if not fields.get("survey_number"):
        warnings.append("Survey number not extracted")

    if not fields.get("hissa"):
        warnings.append("Hissa not extracted")

    if not fields.get("owner_names"):
        warnings.append("Owner names not extracted")

    if not fields.get("extent"):
        warnings.append("Extent not extracted")

    confidence_score = 100

    confidence_score -= len(warnings) * 10

    if confidence_score < 0:
        confidence_score = 0

    return {
        "validated": True,
        "confidence_score": confidence_score,
        "fields": fields,
        "warnings": warnings,
        "requires_manual_review": confidence_score < 80
    }
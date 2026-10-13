"""Map extractor evidence to the public schema without hiding mapping problems."""
from collections.abc import Mapping
from pydantic import ValidationError
from app.schemas.mr import MutationRegister, MutationEntry, SurveyDetail


def map_mr_fields(extracted: Mapping | None, warnings: list[str] | None = None) -> MutationRegister:
    warnings = warnings if warnings is not None else []
    if not isinstance(extracted, Mapping):
        warnings.append("No usable MR extraction was returned.")
        extracted = {}

    def validated(model, raw, path):
        # Metadata is stripped explicitly; unknown nested keys are reported.
        if not isinstance(raw, Mapping):
            warnings.append(f"{path}: expected an object; inspect extraction evidence.")
            raw = {}
        values = dict(raw)
        for key in set(values) - set(model.model_fields):
            warnings.append(f"{path}.{key}: unsupported field retained in extraction evidence.")
            values.pop(key)
        # Validate one field at a time; retain the rest of a partly readable row.
        clean = {}
        for key, value in values.items():
            try:
                clean[key] = getattr(model.model_validate({key: value}), key)
            except ValidationError:
                warnings.append(f"{path}.{key}: invalid value retained in extraction evidence.")
        return model.model_validate(clean)

    nested = {"current_survey_details", "new_survey_details", "affected_survey_numbers", "entries"}
    result = validated(MutationRegister, {
        k: v for k, v in extracted.items() if k in MutationRegister.model_fields and k not in nested
    }, "mr")

    for field in ("current_survey_details", "new_survey_details"):
        raw = extracted.get(field)
        if raw is None and field == "current_survey_details":
            old = extracted.get("survey_details")
            raw = [old] if isinstance(old, Mapping) and any(old.values()) else []
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            warnings.append(f"{field}: expected a list; inspect extraction evidence.")
            raw = []
        setattr(result, field, [validated(SurveyDetail, item, f"{field}[{i}]") for i, item in enumerate(raw)])

    surveys = extracted.get("affected_survey_numbers") or []
    if isinstance(surveys, str):
        surveys = [surveys]
        warnings.append("affected_survey_numbers: legacy string converted to a one-item list.")
    if not isinstance(surveys, list):
        surveys = []
        warnings.append("affected_survey_numbers: invalid list; inspect extraction evidence.")
    for survey in surveys:
        if isinstance(survey, str) and survey.strip() not in {"", "-", "—", "–"}:
            if survey.strip() not in result.affected_survey_numbers:
                result.affected_survey_numbers.append(survey.strip())
        elif survey is not None:
            warnings.append("affected_survey_numbers: invalid item retained in extraction evidence.")

    rows = extracted.get("entries") or []
    if not isinstance(rows, list):
        warnings.append("entries: expected a list; inspect extraction evidence.")
        rows = []
    for i, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            warnings.append(f"entries[{i}]: invalid row retained in extraction evidence.")
            continue
        if any(k in raw for k in ("rights_changed_from", "rights_changed_to", "extent_from", "extent_to")):
            warnings.append(f"entries[{i}]: legacy party layout; reprocess the original document to verify roles and joint holders.")
            sides = [("rights_changed_from", "extent_from", "transferor"), ("rights_changed_to", "extent_to", "transferee")]
            populated = [(name, extent, role) for name, extent, role in sides if raw.get(name) or raw.get(extent)]
            if not populated:
                populated = [("rights_changed_from", "extent_from", "unknown")]
            for name, extent, role in populated:
                # A shared legacy annotation cannot safely be assigned to both parties.
                common = {k: v for k, v in raw.items() if k in MutationEntry.model_fields}
                if len(populated) > 1:
                    for key in ("relation", "khata_number", "possession_mode", "rights", "liabilities", "row_status"):
                        if common.get(key):
                            warnings.append(f"entries[{i}].{key}: side is ambiguous; retained in extraction evidence.")
                        common.pop(key, None)
                common.update(party_name=raw.get(name), extent=raw.get(extent), party_role=role)
                result.entries.append(validated(MutationEntry, common, f"entries[{i}]"))
        else:
            result.entries.append(validated(MutationEntry, raw, f"entries[{i}]"))

    # Add actually observed row codes; never assign document-wide codes to rows.
    for entry in result.entries:
        code = entry.survey_hissa_number
        if code and code not in result.affected_survey_numbers:
            result.affected_survey_numbers.append(code)
    return result


def mr_requires_review(evidence: Mapping | None, mapped: MutationRegister, warnings: list[str]) -> bool:
    evidence = evidence or {}
    verification = evidence.get("text_verification") or {}
    return bool(
        warnings or not mapped.entries or not mapped.mr_number
        or any(not entry.party_name or entry.party_role == "unknown" for entry in mapped.entries)
        or evidence.get("error")
        or evidence.get("mapping_warnings")
        or evidence.get("extraction_status") != "extracted"
        or verification.get("status") in {"mismatch", "disagreement", "inconclusive"}
        or evidence.get("header_disagreements") or evidence.get("entry_disagreements")
        or evidence.get("survey_format_warnings")
    )

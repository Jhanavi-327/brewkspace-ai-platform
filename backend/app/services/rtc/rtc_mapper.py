"""Map legacy numbered RTC blocks to the public Pydantic contract."""
from collections.abc import Mapping
from app.schemas.rtc import (
    RTCDocument, KhataVara, LandRevenue, IrrigationDetails,
    OwnershipDetail, OwnerPossession, LandUseAndCropDetails,
)


def _text(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Expected an RTC scalar value")
    text = str(value).strip()
    if text.casefold() in {"", "null", "none", "n/a", "-", "--", "—", "–"}:
        return None
    return text


def _block(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Expected an RTC object")
    return value


def _rows(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected an RTC row list")
    return [_block(row) for row in value]


def _strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("Expected text or a text list")
    return [text for item in value if (text := _text(item)) is not None]


def _pick(value, fields):
    value = _block(value)
    return {field: _text(value.get(field)) for field in fields}


def map_rtc_fields(extracted: Mapping) -> RTCDocument:
    extracted = _block(extracted)
    data = _pick(extracted.get("header"), (
        "taluk", "hobli", "village", "valid_from", "valid_to"
    ))
    data["survey_number"] = _text(_block(extracted.get("1_survey_number")).get("survey_number"))
    data["hissa_number"] = _text(_block(extracted.get("2_hissa")).get("hissa_number"))
    for target, source, model in (
        ("khata_vara", "3_khata_vara", KhataVara),
        ("land_revenue", "4_land_revenue", LandRevenue),
        ("irrigation_details", "8_irrigation_details", IrrigationDetails),
    ):
        data[target] = _pick(extracted.get(source), model.model_fields)
    data["soil_type"] = _text(_block(extracted.get("5_soil_type")).get("soil_type"))
    data["land_classification"] = _text(_block(extracted.get("6_land_classification")).get("land_classification"))
    trees = _block(extracted.get("7_tree_details"))
    data["tree_details"] = {
        "tree_count": _text(trees.get("tree_count")),
        "tree_types": _strings(trees.get("tree_types")),
    }
    data["ownership_details"] = []
    for row in _rows(extracted.get("9_ownership_details")):
        owner = _pick(row, (
            "owner_name", "father_or_spouse_name", "address", "extent",
            "khata_number", "possession_type",
        ))
        owner["khata_number"] = _text(row.get("khata_number") or row.get("khata_no"))
        mutations = [
            _pick(entry, ("mr_number", "recorded_on", "status", "source_column"))
            for entry in _rows(row.get("mutation_entries"))
        ]
        mutations = [entry for entry in mutations if any(v is not None for v in entry.values())]
        rights = _block(row.get("rights_liabilities"))
        owner_rights = _pick(rights, (
            "court_reference", "rights", "liabilities", "remarks"
        ))
        stay = rights.get("court_stay")
        if stay is not None and not isinstance(stay, bool):
            raise ValueError("Owner court_stay must be a boolean or null")
        owner_rights["court_stay"] = stay
        if (
            any(value is not None for value in owner.values())
            or mutations
            or any(value is not None for value in owner_rights.values())
        ):
            owner["mutation_entries"] = mutations
            owner["rights_liabilities"] = owner_rights
            data["ownership_details"].append(owner)
    # Never infer owner associations by splitting the legacy global blocks.
    # Those blocks remain available in rtc_extraction_details for review.
    data["cultivation_details"] = []
    for row in _rows(extracted.get("12_cultivation_details")):
        crop = _pick(row, (
            "crop_year",
            "season",
            "cultivator_name",
            "cultivation_method",
            "land_use",
            "land_category",
            "crop_name",
            "crop_extent",
            "irrigated_extent",
        ))
        # Keep section 13 data on the SAME row, avoiding ambiguous joins
        # between repeated cultivators and seasons.
        crop_details = _pick(row, LandUseAndCropDetails.model_fields)
        if any(v is not None for v in (*crop.values(), *crop_details.values())):
            crop["land_use_and_crop_details"] = crop_details
            data["cultivation_details"].append(crop)
    return RTCDocument.model_validate(data)


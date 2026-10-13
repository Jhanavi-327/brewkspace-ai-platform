"""Public MR extraction contract. Printed values are strings, not measurements."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MRBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def blank_text(cls, value):
        if isinstance(value, str) and value.strip() in {"", "-", "—", "–"}:
            return None
        return value


class SurveyDetail(MRBase):
    survey_number: str | None = None
    total_extent: str | None = None
    kharab_a: str | None = None
    kharab_b: str | None = None
    revenue: str | None = None
    judi: str | None = None
    cess: str | None = None
    water_rate: str | None = None
    patta: str | None = None
    soil_type: str | None = None


class MutationEntry(MRBase):
    # One party block, in page order. Separate parties are never paired by index.
    source_page: int | None = Field(default=None, ge=1)
    survey_hissa_number: str | None = None
    patta_number: str | None = None
    revenue: str | None = None
    party_role: Literal["transferor", "transferee", "recorded_party", "holder", "unknown"] = "unknown"
    party_name: str | None = None
    extent: str | None = None
    extent_note: str | None = Field(
        default=None,
        description="Printed joint-extent marker, e.g. ಮೇಲಿನ ಜಂಟಿ; not a numeric area.",
    )
    relation: str | None = None
    khata_number: str | None = None
    possession_mode: str | None = None
    rights: str | None = None
    liabilities: str | None = None
    row_status: str | None = None
    inspecting_officer: str | None = None


class MutationRegister(MRBase):
    document_type: Literal["MUTATION_REGISTER"] = "MUTATION_REGISTER"
    namune: str | None = None
    district: str | None = None
    taluk: str | None = None
    hobli: str | None = None
    village: str | None = None
    disposal_year: str | None = None
    disposal_number: str | None = None
    mr_number: str | None = None
    source: str | None = None
    source_number: str | None = None
    change_type: str | None = None
    possession_type: str | None = None
    order_reference_date: str | None = None
    status: str | None = None
    # Separate lists support subdivision records without losing new parcels.
    current_survey_details: list[SurveyDetail] = Field(default_factory=list)
    new_survey_details: list[SurveyDetail] = Field(default_factory=list)
    affected_survey_numbers: list[str] = Field(default_factory=list)
    entries: list[MutationEntry] = Field(default_factory=list)
    mutation_order_details: str | None = None
    place: str | None = None
    order_date: str | None = None
    signatory: str | None = None


# Report layout ("4. MUTATION REGISTER"): one numbered item per mutation.
class RightsParty(MRBase):
    name: str | None = None
    area: str | None = None


class MutationRegisterReportItem(MRBase):
    district: str | None = None
    taluk: str | None = None
    hobli: str | None = None
    village: str | None = None
    transaction_year: str | None = None
    transaction_number: str | None = None
    mutation_register_number: str | None = None
    source: str | None = None
    type_of_change: str | None = Field(
        default=None,
        description="e.g. Tiluvalike Chethi, Khatha Badalavane, Hakku and Runa, Podi.",
    )
    mode_of_acquisition: str | None = None
    number: str | None = None
    date: str | None = None
    affected_survey_number_and_hissa: list[str] = Field(default_factory=list)
    # Each party carries its own area; lists keep joint holders separate.
    transferors_of_rights: list[RightsParty] = Field(default_factory=list)
    transferees_of_rights: list[RightsParty] = Field(default_factory=list)
    revenue_officer_remarks: str | None = None
    mutation_order_details: str | None = None
    revenue_inspector_order_date: str | None = None
    additional_information: str | None = None
    place: str | None = None
    mutation_order_date: str | None = None


class MutationRegisterReport(MRBase):
    items: list[MutationRegisterReportItem] = Field(default_factory=list)

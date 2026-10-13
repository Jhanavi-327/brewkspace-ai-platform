"""Public RTC fields. Preserve printed extents, dates and Kannada text."""
from pydantic import BaseModel, ConfigDict, Field


class RTCBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class KhataVara(RTCBase):
    total_extent: str | None = None
    phut_kharab_a: str | None = None
    phut_kharab_b: str | None = None
    remaining_extent: str | None = None


class LandRevenue(RTCBase):
    land_revenue: str | None = None
    jodi: str | None = None
    cess: str | None = None
    water_rate: str | None = None
    total_revenue: str | None = None


class IrrigationDetails(RTCBase):
    source: str | None = None
    kharif_extent: str | None = None
    rabi_extent: str | None = None
    garden_extent: str | None = None
    total_extent: str | None = None


class MutationEntry(RTCBase):
    mr_number: str | None = None
    recorded_on: str | None = None
    status: str | None = None
    source_column: str | None = None


class OwnerRightsLiabilities(RTCBase):
    court_stay: bool | None = None
    court_reference: str | None = None
    rights: str | None = None
    liabilities: str | None = None
    remarks: str | None = None


class OwnershipDetail(RTCBase):
    owner_name: str | None = None
    father_or_spouse_name: str | None = None
    address: str | None = None
    extent: str | None = None
    khata_number: str | None = None
    possession_type: str | None = None
    mutation_entries: list[MutationEntry] = Field(default_factory=list)
    rights_liabilities: OwnerRightsLiabilities = Field(default_factory=OwnerRightsLiabilities)


class OwnerPossession(RTCBase):
    owner_name: str | None = None
    possession_type: str | None = None


class LandUseAndCropDetails(RTCBase):
    dry_land_crop_area: str | None = None
    wet_irrigated_land_crop_area: str | None = None
    garden_land_crop_area: str | None = None
    source_of_irrigation: str | None = None
    crop_yield: str | None = None
    mixed_crop_name: str | None = None
    mixed_crop_area: str | None = None


class TreeDetails(RTCBase):
    tree_count: str | None = None
    tree_types: list[str] = Field(default_factory=list)


class PossessionDetails(RTCBase):
    possession_type: list[str] = Field(default_factory=list)
    owner_wise_possession: list[OwnerPossession] = Field(default_factory=list)


class RightsLiabilities(RTCBase):
    mr_number: str | None = None
    court_stay: bool | None = None
    court_reference: str | None = None
    rights: str | None = None
    liabilities: str | None = None
    remarks: str | None = None


class CultivationDetail(RTCBase):
    crop_year: str | None = None
    season: str | None = None
    cultivator_name: str | None = None
    cultivation_method: str | None = None
    land_use: str | None = None
    land_category: str | None = None
    crop_name: str | None = None
    crop_extent: str | None = None
    irrigated_extent: str | None = None
    land_use_and_crop_details: LandUseAndCropDetails = Field(
        default_factory=LandUseAndCropDetails
    )


class RTCDocument(RTCBase):
    taluk: str | None = None
    hobli: str | None = None
    village: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    survey_number: str | None = None
    hissa_number: str | None = None
    khata_vara: KhataVara = Field(default_factory=KhataVara)
    land_revenue: LandRevenue = Field(default_factory=LandRevenue)
    soil_type: str | None = None
    land_classification: str | None = None
    tree_details: TreeDetails = Field(default_factory=TreeDetails)
    irrigation_details: IrrigationDetails = Field(default_factory=IrrigationDetails)
    ownership_details: list[OwnershipDetail] = Field(default_factory=list)
    cultivation_details: list[CultivationDetail] = Field(default_factory=list)

"""Public Patta Book (Khata extract) contract. Printed values are strings, not measurements."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PattaBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def blank_text(cls, value):
        if isinstance(value, str) and value.strip() in {"", "-", "—", "–"}:
            return None
        return value


class PattaAmounts(PattaBase):
    # Extent is printed as acre.gunta.anna.pai, e.g. 0.22.00.00; keep as printed.
    extent: str | None = Field(default=None, description="ವಿಸ್ತೀರ್ಣ, e.g. 0.22.00.00")
    akara: str | None = Field(default=None, description="ಆಕಾರ")
    local_cess: str | None = Field(default=None, description="ಸ್ಥಳೀಯ ಕರ")
    health_cess: str | None = Field(default=None, description="ಆರೋಗ್ಯ ಕರ")
    education_cess: str | None = Field(default=None, description="ವಿದ್ಯಾ ಕರ")
    total: str | None = Field(default=None, description="ಒಟ್ಟು")


class PattaEntry(PattaAmounts):
    serial_number: str | None = Field(default=None, description="ಕ್ರಮ ಸಂಖ್ಯೆ")
    survey_hissa_number: str | None = Field(default=None, description="ಸರ್ವೆ ಮತ್ತು ಹಿಸ್ಸಾ ನಂ, e.g. 28/*/10")
    # Multi-line cells (joint holders, "ಬಿನ್ ಲೇಟ್ ...") stay in one printed string.
    holder_name: str | None = Field(default=None, description="ಸ್ವಾಧೀನದಾರರ ಹೆಸರು")
    father_name: str | None = Field(default=None, description="ತಂದೆಯ ಹೆಸರು, only when printed separately")
    numeric_check: Literal["ok", "mismatch", "unknown"] | None = Field(
        default=None,
        description="Extractor check that akara + cesses == total for this row.",
    )


class PattaBook(PattaBase):
    document_type: Literal["PATTA_BOOK"] = "PATTA_BOOK"
    khata_number: str | None = Field(default=None, description="ಖಾತೆ ನಂಬರ್")
    district: str | None = None
    taluk: str | None = None
    hobli: str | None = None
    village: str | None = None
    entries: list[PattaEntry] = Field(default_factory=list)
    total_row: PattaAmounts = Field(default_factory=PattaAmounts, description="Printed ಒಟ್ಟು row")
    place: str | None = Field(default=None, description="ಮೊಕ್ಕಾಂ")
    date: str | None = Field(default=None, description="ದಿನಾಂಕ, as printed (M/D/YYYY)")
    amount: str | None = Field(default=None, description="ಮೊತ್ತ, e.g. 0")
    is_certified_copy: bool | None = Field(default=None, description="ದೃಢೀಕೃತ ಪ್ರತಿ footer present")

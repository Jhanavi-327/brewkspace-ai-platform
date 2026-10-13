from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PanCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pan_number: str | None = Field(
        ...,
        pattern=r"^[A-Z]{5}[0-9]{4}[A-Z]$",
        description="Ten-character PAN number, in uppercase.",
    )

    name: str | None = Field(
        ...,
        min_length=1,
        description="Cardholder's name as printed on the PAN card.",
    )

    father_name: str | None = Field(
        ...,
        min_length=1,
        description="Father's name as printed on the PAN card.",
    )

    date_of_birth: str | None = Field(
        ...,
        pattern=r"^[0-9]{2}/[0-9]{2}/[0-9]{4}$",
        description="Date of birth in DD/MM/YYYY format.",
    )

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, value: str | None) -> str | None:
        if value is not None:
            # Reject impossible dates such as 31/02/1990.
            datetime.strptime(value, "%d/%m/%Y")
        return value
from pydantic import BaseModel, ConfigDict, Field


class AadhaarCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str | None = Field(
        ...,
        min_length=1,
        description="Cardholder's name extracted from the document.",
    )

    date_of_birth_or_year: str | None = Field(
        ...,
        pattern=r"^(?:[0-9]{4}|[0-9]{2}/[0-9]{2}/[0-9]{4})$",
        description=(
            "Date of birth in DD/MM/YYYY format, or YYYY if only "
            "the birth year is printed. Never invent a day or month."
        ),
    )

    gender: str | None = Field(
        ...,
        min_length=1,
        description="Gender extracted from the document.",
    )

    aadhaar_number: str | None = Field(
        ...,
        min_length=1,
        description=(
            "Aadhaar number as extracted, preserving spaces "
            "and any masking characters."
        ),
    )

    address: str | None = Field(
        ...,
        min_length=1,
        description="Complete extracted address, including PIN code.",
    )
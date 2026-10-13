from pydantic import BaseModel
from typing import Optional


class InvestigationCreate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None


class InvestigationResponse(BaseModel):
    id: str
    name: Optional[str]
    notes: Optional[str]
    status: str
    organisation_id: str
    user_id: str

    class Config:
        from_attributes = True
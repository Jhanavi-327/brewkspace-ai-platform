from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, JSON
from sqlalchemy.sql import func

from app.core.database import Base


class PropertyRecord(Base):
    __tablename__ = "property_records"

    id = Column(String, primary_key=True)
    investigation_id = Column(String, ForeignKey("investigations.id"), nullable=False)
    organisation_id = Column(String, ForeignKey("organisations.id"), nullable=False)

    district = Column(String, nullable=True)
    taluk = Column(String, nullable=True)
    hobli = Column(String, nullable=True)
    village = Column(String, nullable=True)
    survey_number = Column(String, nullable=True)
    property_number = Column(String, nullable=True)

    fields = Column(JSON, nullable=True)

    completeness_percentage = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
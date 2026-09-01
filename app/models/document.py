from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Float, JSON
from sqlalchemy.sql import func

from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)

    investigation_id = Column(
        String,
        ForeignKey("investigations.id"),
        nullable=False
    )

    organisation_id = Column(
        String,
        ForeignKey("organisations.id"),
        nullable=False
    )

    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    storage_path = Column(String, nullable=False)

    classification = Column(String, nullable=True)
    classification_confidence = Column(Float, nullable=True)

    extracted_fields = Column(JSON, nullable=True)

    processing_status = Column(String, default="UPLOADED")

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
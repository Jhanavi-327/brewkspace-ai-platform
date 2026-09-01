from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.core.database import Base


class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(String, primary_key=True)
    organisation_id = Column(String, ForeignKey("organisations.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    name = Column(String, nullable=True)
    notes = Column(String, nullable=True)

    status = Column(String, default="UPLOADED")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
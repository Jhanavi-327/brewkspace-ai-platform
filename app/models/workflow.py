from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func

from app.core.database import Base


class WorkflowStatusLog(Base):
    __tablename__ = "workflow_status_logs"

    id = Column(String, primary_key=True)
    investigation_id = Column(String, ForeignKey("investigations.id"), nullable=False)
    organisation_id = Column(String, ForeignKey("organisations.id"), nullable=False)

    current_state = Column(String, default="UPLOADED")
    stages = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import DateTime
from sqlalchemy.sql import func

from app.core.database import Base


class Organisation(Base):
    __tablename__ = "organisations"

    id = Column(String, primary_key=True)

    name = Column(String, nullable=False)

    subscription_tier = Column(
        String,
        default="Starter"
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
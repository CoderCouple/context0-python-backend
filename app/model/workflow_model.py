from datetime import datetime
from uuid import uuid4

from sqlalchemy import TIMESTAMP, Boolean, Column, Enum, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.common.enums import WorkflowStatus
from app.db.base import Base


class Workflow(Base):
    __tablename__ = "workflow"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)

    name = Column(String(255))
    description = Column(String(1024))
    definition = Column(Text)
    execution_plan = Column(Text)
    cron = Column(String(100))
    status = Column(Enum(WorkflowStatus), nullable=False)

    credits_cost = Column(Numeric)
    last_run_at = Column(TIMESTAMP(timezone=True))
    last_run_id = Column(UUID(as_uuid=True))
    last_run_status = Column(String(50))
    next_run_at = Column(TIMESTAMP(timezone=True))

    created_by = Column(UUID(as_uuid=True), nullable=False)
    updated_by = Column(UUID(as_uuid=True), nullable=False)

    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )

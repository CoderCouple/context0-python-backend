from datetime import datetime
from decimal import Decimal
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from app.common.enum.workflow import WorkflowStatus


class CreateWorkflowRequest(BaseModel):
    name: Annotated[str, Field(max_length=255)]
    description: Optional[Annotated[str, Field(max_length=1024)]] = None
    status: Annotated[str, Field(max_length=50)] = WorkflowStatus.DRAFT


class UpdateWorkflowRequest(BaseModel):
    workflow_id: str
    name: Optional[Annotated[str, Field(max_length=255)]] = None
    description: Optional[Annotated[str, Field(max_length=1024)]] = None
    definition: Optional[dict] = None
    execution_plan: Optional[str] = None
    cron: Optional[Annotated[str, Field(max_length=100)]] = None
    status: Optional[Annotated[str, Field(max_length=50)]] = None
    credits_cost: Optional[
        Annotated[Decimal, Field(max_digits=10, decimal_places=2)]
    ] = None
    last_run_at: Optional[datetime] = None
    last_run_id: Optional[str] = None
    last_run_status: Optional[Annotated[str, Field(max_length=50)]] = None
    next_run_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class DeleteWorkflowRequest(BaseModel):
    workflow_id: str
    is_soft_delete: bool = True

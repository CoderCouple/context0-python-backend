from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, RootModel


class WorkflowExecutionResponse(BaseModel):
    id: str
    workflow_id: str
    user_id: str
    trigger: str
    status: str
    credits_consumed: Optional[Decimal]

    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    created_by: str
    updated_by: str
    updated_at: datetime
    is_deleted: bool

    model_config = {"from_attributes": True}


class WorkflowExecutionListResponse(RootModel[List[WorkflowExecutionResponse]]):
    @classmethod
    def from_orm_list(cls, executions: list) -> "WorkflowExecutionListResponse":
        return cls([WorkflowExecutionResponse.model_validate(e) for e in executions])


class ExecutionLogResponse(BaseModel):
    id: str
    log_level: str
    message: str
    timestamp: datetime
    created_by: str
    updated_by: str
    updated_at: datetime
    created_at: datetime
    is_deleted: bool

    model_config = {"from_attributes": True}


class ExecutionPhaseResponse(BaseModel):
    id: str
    user_id: str
    workflow_execution_id: str
    status: str
    number: int
    node: Optional[str]
    name: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    inputs: Optional[str]
    outputs: Optional[str]
    credits_consumed: Optional[Decimal]
    created_by: str
    updated_by: str
    updated_at: datetime
    created_at: datetime
    is_deleted: bool

    logs: List[ExecutionLogResponse] = []

    model_config = {"from_attributes": True}


class WorkflowExecutionWithPhasesResponse(WorkflowExecutionResponse):
    phases: List[ExecutionPhaseResponse]

    model_config = {"from_attributes": True}


class WorkflowPhaseWithLogsResponse(ExecutionPhaseResponse):
    logs: List[ExecutionLogResponse]

    model_config = {"from_attributes": True}

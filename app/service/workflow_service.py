from datetime import datetime
from uuid import uuid4

from fastapi import HTTPException
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate
from sqlalchemy.orm import Session

from app.api.v1.request.workflow_request import (
    CreateWorkflowRequest,
    UpdateWorkflowRequest,
)
from app.api.v1.response.workflow_response import WorkflowListResponse, WorkflowResponse
from app.common.auth import UserContext
from app.common.enums import WorkflowStatus
from app.common.validation.workflow_definition_schema import workflow_definition_schema
from app.model.workflow_model import Workflow


class WorkflowService:
    def __init__(self, db: Session, context: UserContext):
        self.db = db
        self.context = context

    def fetch_workflow_by_id(self, workflow_id: str) -> WorkflowResponse:
        workflow = (
            self.db.query(Workflow)
            .filter(Workflow.id == workflow_id, not Workflow.is_deleted)
            .first()
        )

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Optional org-level access check
        # if workflow.organization_id != self.context.organization_id:
        #     raise HTTPException(status_code=403, detail="Access denied")

        return WorkflowResponse.from_orm(workflow)

    def fetch_workflows(self) -> WorkflowListResponse:
        workflows = (
            self.db.query(Workflow)
            .filter(
                Workflow.user_id == self.context.user_id,
                not Workflow.is_deleted,
            )
            .all()
        )

        if not workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return WorkflowListResponse.from_orm_list(workflows)

    def create_workflow(self, body: CreateWorkflowRequest) -> WorkflowResponse:
        if body.status not in [
            workflow_status.value for workflow_status in WorkflowStatus
        ]:
            raise HTTPException(status_code=400, detail="Invalid status")

        workflow = Workflow(
            id=f"workflow_{uuid4()}",
            user_id=self.context.user_id,
            # organization_id=self.context.organization_id,
            created_by=self.context.user_id,
            updated_by=self.context.user_id,
            **body.dict(),
        )
        self.db.add(workflow)
        self.db.commit()
        self.db.refresh(workflow)

        return WorkflowResponse.from_orm(workflow)

    def delete_workflow(
        self, workflow_id: str, is_soft_delete: bool
    ) -> WorkflowResponse:
        workflow = self.db.query(Workflow).filter_by(id=workflow_id).first()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # if workflow.organization_id != self.context.organization_id:
        #     raise HTTPException(
        #         status_code=403,
        #         detail="Access denied: not your organization"
        #     )

        if is_soft_delete:
            workflow.is_deleted = True
            workflow.updated_by = self.context.user_id
            workflow.updated_at = datetime.utcnow()
        else:
            self.db.delete(workflow)

        self.db.commit()
        return WorkflowResponse.from_orm(workflow)

    def update_workflow(self, body: UpdateWorkflowRequest) -> WorkflowResponse:
        workflow = (
            self.db.query(Workflow)
            .filter(Workflow.id == body.workflow_id, not Workflow.is_deleted)
            .first()
        )

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if not workflow.status == WorkflowStatus.DRAFT:
            raise HTTPException(
                status_code=409, detail="Workflow is already published!"
            )

        if body.definition is not None:
            try:
                validate(instance=body.definition, schema=workflow_definition_schema)
            except ValidationError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid workflow definition: {e.message}"
                )
            workflow.definition = body.definition

        # Optional org-level check
        # if workflow.organization_id != self.context.organization_id:
        #     raise HTTPException(status_code=403, detail="Access denied")

        if body.name is not None:
            workflow.name = body.name
        if body.description is not None:
            workflow.description = body.description
        if body.definition is not None:
            workflow.definition = body.definition
        if body.execution_plan is not None:
            workflow.execution_plan = body.execution_plan
        if body.cron is not None:
            workflow.cron = body.cron
        if body.status is not None:
            workflow.status = body.status
        if body.credits_cost is not None:
            workflow.credits_cost = body.credits_cost
        if body.next_run_at is not None:
            workflow.next_run_at = body.next_run_at

        workflow.updated_by = self.context.user_id
        workflow.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(workflow)

        return WorkflowResponse.from_orm(workflow)

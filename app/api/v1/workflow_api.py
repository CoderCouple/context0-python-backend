"""Workflow API."""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.base_response import BaseResponse, success_response
from app.api.tags import Tags
from app.api.v1.request.workflow_request import (
    CreateWorkflowRequest,
    DeleteWorkflowRequest,
    UpdateWorkflowRequest,
)
from app.api.v1.response.workflow_response import WorkflowListResponse, WorkflowResponse
from app.common.auth import UserContext
from app.common.role_decorator import require_roles
from app.db.session import get_db
from app.service.workflow_service import WorkflowService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.workflow])


@router.get("/workflow", response_model=BaseResponse[WorkflowListResponse])
def get_workflows(
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    workflows = WorkflowService(db, context).fetch_workflows()
    return success_response(workflows, "Workflows fetched successfully")


@router.get("/workflow/{workflow_id}", response_model=BaseResponse[WorkflowResponse])
def get_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    workflow = WorkflowService(db, context).fetch_workflow_by_id(workflow_id)
    return success_response(workflow, "Workflow fetched successfully")


@router.post("/workflow", response_model=BaseResponse[WorkflowResponse])
def create_workflow(
    body: CreateWorkflowRequest,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    workflow = WorkflowService(db, context).create_workflow(body)
    return success_response(workflow, "Workflow created successfully", 200)


@router.put("/workflow", response_model=BaseResponse[WorkflowResponse])
def update_workflow(
    body: UpdateWorkflowRequest,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    workflow = WorkflowService(db, context).update_workflow(body)
    return success_response(workflow, "Workflow updated successfully", 200)


@router.delete("/workflow", response_model=BaseResponse[WorkflowResponse])
def delete_workflow(
    body: DeleteWorkflowRequest,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    workflow = WorkflowService(db, context).delete_workflow(
        workflow_id=body.workflow_id, is_soft_delete=body.is_soft_delete
    )
    return success_response(workflow, "Workflow deleted successfully", 200)

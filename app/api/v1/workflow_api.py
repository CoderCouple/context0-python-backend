"""Workflow API."""
import logging
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.base_response import BaseResponse, success_response
from app.api.tags import Tags
from app.api.v1.request.workflow_request import CreateWorkflowRequest
from app.api.v1.response.workflow_response import WorkflowListResponse, WorkflowResponse
from app.core.auth import UserContext
from app.core.role_decorator import require_roles
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
    workflow_id: UUID,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    workflow = WorkflowService(db, context).fetch_workflow_by_id(workflow_id)
    return success_response(workflow, "Workflow fetched successfully")


@router.post("/workflow", response_model=BaseResponse[WorkflowResponse])
def create_workflow_route(
    body: CreateWorkflowRequest,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    workflow = WorkflowService(db, context).create_workflow(body)
    return success_response(workflow, "Workflow created successfully", 200)

from fastapi import APIRouter

from app.api.v1.ping_api import router as hello_api_router
from app.api.v1.workflow_api import router as workflow_api_router

router = APIRouter(prefix="/v1")

# Mount feature routers
router.include_router(hello_api_router)
router.include_router(workflow_api_router)

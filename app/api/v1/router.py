from fastapi import APIRouter

from app.api.v1.controller.analytics_api import router as analytics_api_router
from app.api.v1.controller.billing_api import router as billing_api_router
from app.api.v1.controller.chat_api import router as chat_api_router
from app.api.v1.controller.credential_api import router as credential_api_router
from app.api.v1.controller.llm_config_api import router as llm_config_api_router
from app.api.v1.controller.llm_memory_api import router as llm_memory_api_router
from app.api.v1.controller.memory_api import router as memory_api_router
from app.api.v1.controller.ping_api import router as ping_api_router
from app.api.v1.controller.qa_api import router as qa_api_router
from app.api.v1.controller.webhook_api import router as webhook_api_router
from app.api.v1.controller.metrics_api import router as metrics_api_router

# from app.api.v1.controller.workflow_api import router as workflow_api_router  # Disabled - workflow models deleted

router = APIRouter(prefix="/v1")

# Mount feature routers
router.include_router(ping_api_router)
router.include_router(memory_api_router)
router.include_router(llm_memory_api_router)
router.include_router(llm_config_api_router)
router.include_router(qa_api_router)
router.include_router(chat_api_router)
# router.include_router(workflow_api_router)  # Disabled - workflow models deleted
router.include_router(analytics_api_router)
router.include_router(billing_api_router)
router.include_router(credential_api_router)
router.include_router(webhook_api_router)
router.include_router(metrics_api_router)

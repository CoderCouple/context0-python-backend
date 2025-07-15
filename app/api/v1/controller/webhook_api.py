"""Webhook API."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from svix.webhooks import Webhook, WebhookVerificationError

from app.api.tags import Tags
from app.common.auth.auth import UserContext
from app.db.session import get_db
from app.service.webhook_service import WebhookService
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Webhook])

CLERK_WEBHOOK_SIGNING_SECRET = settings.clerk_webhook_signing_secret


@router.post("/webhook/clerk", status_code=status.HTTP_204_NO_CONTENT)
async def handle_clerk_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    payload = await request.body()
    print("DEBUG EVENT:", payload)
    # return {"status": "ok"}
    headers = request.headers

    wh = Webhook(CLERK_WEBHOOK_SIGNING_SECRET)

    try:
        event = wh.verify(payload, headers)
    except WebhookVerificationError as e:
        logger.warning("Webhook verification failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid signature") from e

    event_type = event["type"]
    data = event["data"]

    logger.info("Received Clerk webhook type: %s", event_type)
    logger.info("Received Clerk webhook data: %s", data)

    # Dummy system-level context (used only if your WebhookService requires it)
    context = UserContext(user_id="clerk_system", role="admin")

    WebhookService(db, context).handle_event(event_type, data)

    return

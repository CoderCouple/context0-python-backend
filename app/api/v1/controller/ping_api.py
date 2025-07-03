""" Ping API."""

import logging

from fastapi import APIRouter

from app.api.tags import Tags

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Ping])


@router.get("/")
def ping():
    return {
        "status": "OK",
        "message": "Context0 Backend API is running. Alive and healthy!",
    }

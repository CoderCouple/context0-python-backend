import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuthLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user_id = request.headers.get("X-User-Id")
        org_id = request.headers.get("X-Org-Id")
        role = request.headers.get("X-Role")

        logger.info(f"Request from user_id={user_id}, org_id={org_id}, role={role}")

        response = await call_next(request)
        return response

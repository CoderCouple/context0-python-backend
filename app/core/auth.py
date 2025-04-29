import os
from typing import Optional

from clerk_backend_api import Clerk
from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from app.settings import settings

# Patch env for clerk SDK
os.environ["CLERK_API_KEY"] = settings.clerk_api_key

clerk = Clerk()


class UserContext(BaseModel):
    user_id: str
    organization_id: Optional[str] = None
    role: Optional[str] = None


async def get_current_user_context(request: Request) -> UserContext:
    if settings.auth_disabled:
        return UserContext(
            user_id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
            organization_id="dev-org-id",
            role="admin",
        )

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    token = auth_header.split("Bearer ")[1].strip()

    try:
        session = clerk.sessions.verify(token)
        if not session or not session.is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session"
            )

        return UserContext(
            user_id=session.user_id,
            organization_id=session.organization_id,
            role=session.organization_role,
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

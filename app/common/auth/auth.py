import os
from typing import Optional

from clerk_backend_api import Clerk
from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from app.settings import settings

# Patch env for clerk SDK
os.environ["CLERK_API_KEY"] = settings.clerk_secret_key

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
        import jwt
        from jwt import PyJWKClient

        # First decode without verification to get the issuer
        unverified = jwt.decode(token, options={"verify_signature": False})

        # Get the issuer to construct the JWKS URL
        # Clerk tokens have issuer like: https://your-domain.clerk.accounts.dev
        issuer = unverified.get("iss")
        if not issuer:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: no issuer found",
            )

        # Clerk's JWKS endpoint is at the issuer URL + /.well-known/jwks.json
        jwks_url = f"{issuer}/.well-known/jwks.json"

        print(f"Fetching JWKS from: {jwks_url}")

        # Get the public keys and verify the token
        jwks_client = PyJWKClient(jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        # Verify and decode the token
        decoded = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=None,  # Clerk tokens might not have audience
            options={"verify_aud": False},
        )

        # Extract user information from verified token
        user_id = decoded.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: no user ID found",
            )

        # Extract organization info if present
        org_id = decoded.get("org_id")
        org_role = decoded.get("org_role")

        # Extract role from metadata if not in org_role
        if not org_role:
            # Try public metadata
            metadata = decoded.get("metadata", {})
            org_role = metadata.get("role", "user")

        print(f"Token verified for user: {user_id}, org: {org_id}, role: {org_role}")

        return UserContext(
            user_id=user_id, organization_id=org_id, role=org_role or "user"
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Token verification error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

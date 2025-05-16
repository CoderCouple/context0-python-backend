from typing import List

from fastapi import Depends, HTTPException, status

from app.common.auth import UserContext, get_current_user_context


def require_roles(roles: List[str]):
    def wrapper(
        context: UserContext = Depends(get_current_user_context),
    ) -> UserContext:
        if context.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of the roles: {roles}",
            )
        return context

    return wrapper

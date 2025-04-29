from typing import Dict, Optional

from fastapi import HTTPException

from app.common.error.http_error_enum import ErrorCode


class AppError(HTTPException):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int,
        extra: Optional[Dict] = None,
    ):
        self.code = code
        self.message = message
        self.extra = extra or {}

        super().__init__(
            status_code=status_code,
            detail={
                "code": code,
                "message": message,
                "extra": self.extra,
            },
        )

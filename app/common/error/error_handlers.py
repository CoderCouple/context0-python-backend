# app/common/error/error_handlers.py

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from app.api.base_response import BaseResponse
from app.common.error.app_error import AppError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI):
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=BaseResponse(
                result=None,
                status_code=exc.status_code,
                message=exc.detail["message"],
                success=False,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            content=BaseResponse(
                result=None,
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                message="Request json_validation failed",
                success=False,
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=BaseResponse(
                result=None,
                status_code=exc.status_code,
                message=str(exc.detail),
                success=False,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content=BaseResponse(
                result=None,
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                message="An unexpected error occurred.",
                success=False,
            ).model_dump(),
        )

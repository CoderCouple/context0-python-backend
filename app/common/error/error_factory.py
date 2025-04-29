# app/common/error/error_factory.py

from fastapi import status

from app.common.error.app_error import AppError
from app.common.error.http_error_enum import ErrorCode


def NotFound(message: str, extra=None):
    return AppError(ErrorCode.NOT_FOUND, message, status.HTTP_404_NOT_FOUND, extra)


def Forbidden(message: str, extra=None):
    return AppError(ErrorCode.FORBIDDEN, message, status.HTTP_403_FORBIDDEN, extra)


def BadRequest(message: str, extra=None):
    return AppError(
        ErrorCode.INVALID_REQUEST, message, status.HTTP_400_BAD_REQUEST, extra
    )


def Unauthorized(message: str, extra=None):
    return AppError(
        ErrorCode.UNAUTHORIZED, message, status.HTTP_401_UNAUTHORIZED, extra
    )


def ServerError(message: str, extra=None):
    return AppError(
        ErrorCode.SERVER_ERROR, message, status.HTTP_500_INTERNAL_SERVER_ERROR, extra
    )

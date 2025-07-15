from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    result: Optional[T] = None
    status_code: int
    message: Optional[str] = None
    success: Optional[bool] = None


def success_response(
    result: Optional[T] = None, message: str = "Success", status_code: int = 200
) -> BaseResponse[T]:
    return BaseResponse(
        result=result,
        status_code=status_code,
        message=message or "Success",
        success=True,
    )


def error_response(
    message: str = "Something went wrong", status_code: int = 500
) -> BaseResponse[None]:
    return BaseResponse(
        result=None,
        status_code=status_code,
        message=message or "Something went wrong",
        success=False,
    )

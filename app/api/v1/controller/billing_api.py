"""Billing API Controller for payment processing and credit management"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session
from starlette import status

from app.api.tags import Tags
from app.api.v1.request.billing_request import PurchaseCreditsRequest
from app.api.v1.response.base_response import (
    BaseResponse,
    error_response,
    success_response,
)
from app.api.v1.response.billing_response import (
    AvailableCreditsResponse,
    BillingHealthResponse,
    CreditPacksResponse,
    DownloadInvoiceResponse,
    PurchaseCreditsResponse,
    SetupUserResponse,
    UserBillingSummary,
    UserPurchase,
)
from app.common.auth.auth import UserContext, get_current_user_context
from app.common.auth.role_decorator import require_roles
from app.db.session import get_db
from app.service.billing_service import BillingService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Billing])


# Dependency injection for billing service
async def get_billing_service() -> BillingService:
    """Get billing service instance"""
    return BillingService()


@router.get("/billing/credits", response_model=BaseResponse[AvailableCreditsResponse])
async def get_available_credits(
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get available credits for a user"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(f"Getting available credits for user {actual_user_id}")

        response = await billing_service.get_available_credits(actual_user_id)

        return success_response(
            result=response,
            message="Available credits retrieved successfully",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Get credits endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving credits",
            status_code=500,
        )


@router.post("/billing/purchase", response_model=BaseResponse[PurchaseCreditsResponse])
async def purchase_credits(
    request: PurchaseCreditsRequest,
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Purchase credits"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(f"Processing credit purchase for user {actual_user_id}")

        response = await billing_service.purchase_credits(actual_user_id, request)

        return success_response(
            result=response,
            message="Credit purchase initiated successfully",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Purchase credits endpoint error: {e}")
        return error_response(
            message="Internal server error processing purchase",
            status_code=500,
        )


@router.post("/billing/setup", response_model=BaseResponse[SetupUserResponse])
async def setup_user(
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    email: Optional[str] = Query(None, description="User email for billing"),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Setup user for billing"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(f"Setting up billing for user {actual_user_id}")

        response = await billing_service.setup_user(actual_user_id, email)

        return success_response(
            result=response,
            message="User billing setup completed successfully",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Setup user endpoint error: {e}")
        return error_response(
            message="Internal server error setting up user",
            status_code=500,
        )


@router.get("/billing/purchases", response_model=BaseResponse[list[UserPurchase]])
async def get_user_purchase_history(
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get user's purchase history"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(f"Getting purchase history for user {actual_user_id}")

        response = await billing_service.get_user_purchase_history(actual_user_id)

        return success_response(
            result=response,
            message="Purchase history retrieved successfully",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Get purchase history endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving purchase history",
            status_code=500,
        )


@router.get(
    "/billing/invoice/{purchase_id}",
    response_model=BaseResponse[DownloadInvoiceResponse],
)
async def download_invoice(
    purchase_id: str = Path(..., description="Purchase ID for the invoice"),
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Download invoice for a purchase"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(f"Generating invoice download for purchase {purchase_id}")

        response = await billing_service.download_invoice(actual_user_id, purchase_id)

        return success_response(
            result=response,
            message="Invoice download URL generated successfully",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Download invoice endpoint error: {e}")
        return error_response(
            message="Internal server error generating invoice",
            status_code=500,
        )


@router.get("/billing/packs", response_model=BaseResponse[CreditPacksResponse])
async def get_credit_packs(
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get available credit packs"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(f"Getting credit packs for user {actual_user_id}")

        response = await billing_service.get_credit_packs(actual_user_id)

        return success_response(
            result=response,
            message="Credit packs retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Get credit packs endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving credit packs",
            status_code=500,
        )


@router.get("/billing/summary", response_model=BaseResponse[UserBillingSummary])
async def get_user_billing_summary(
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get user billing summary"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(f"Getting billing summary for user {actual_user_id}")

        response = await billing_service.get_user_billing_summary(actual_user_id)

        return success_response(
            result=response,
            message="Billing summary retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Get billing summary endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving billing summary",
            status_code=500,
        )


@router.get("/billing/health", response_model=BaseResponse[BillingHealthResponse])
async def get_billing_health(
    billing_service: BillingService = Depends(get_billing_service),
):
    """Get billing system health"""
    try:
        logger.info("Getting billing system health")

        response = await billing_service.get_billing_health()

        return success_response(
            result=response,
            message="Billing system health retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Billing health endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving system health",
            status_code=500,
        )


# Admin endpoints for credit management
@router.post("/billing/admin/add-credits", response_model=BaseResponse[dict])
async def admin_add_credits(
    user_id: str = Query(..., description="User ID to add credits to"),
    credits: int = Query(..., description="Number of credits to add", gt=0),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin"])),
):
    """Admin endpoint to add credits to user account"""
    try:
        logger.info(
            f"Admin {context.user_id} adding {credits} credits to user {user_id}"
        )

        new_total = await billing_service.add_credits(user_id, credits)

        return success_response(
            result={"new_total": new_total, "credits_added": credits},
            message=f"Added {credits} credits successfully",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Admin add credits endpoint error: {e}")
        return error_response(
            message="Internal server error adding credits",
            status_code=500,
        )


@router.post("/billing/admin/deduct-credits", response_model=BaseResponse[dict])
async def admin_deduct_credits(
    user_id: str = Query(..., description="User ID to deduct credits from"),
    credits: int = Query(..., description="Number of credits to deduct", gt=0),
    billing_service: BillingService = Depends(get_billing_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin"])),
):
    """Admin endpoint to deduct credits from user account"""
    try:
        logger.info(
            f"Admin {context.user_id} deducting {credits} credits from user {user_id}"
        )

        new_total = await billing_service.deduct_credits(user_id, credits)

        return success_response(
            result={"new_total": new_total, "credits_deducted": credits},
            message=f"Deducted {credits} credits successfully",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Admin deduct credits endpoint error: {e}")
        return error_response(
            message="Internal server error deducting credits",
            status_code=500,
        )

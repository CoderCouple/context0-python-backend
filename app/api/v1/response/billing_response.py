"""Billing API response models"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.api.v1.request.billing_request import PackId


class AvailableCreditsResponse(BaseModel):
    """Response for available credits"""

    credits: int = Field(..., description="Number of available credits")
    last_updated: datetime = Field(..., description="When credits were last updated")
    expires_at: Optional[datetime] = Field(None, description="When credits expire")


class SetupUserResponse(BaseModel):
    """Response for user setup"""

    user_id: str = Field(..., description="User ID")
    credits: int = Field(..., description="Initial credits assigned")
    billing_setup: bool = Field(..., description="Whether billing is set up")
    customer_id: Optional[str] = Field(None, description="Stripe customer ID")


class PurchaseCreditsResponse(BaseModel):
    """Response for credit purchase"""

    checkout_url: str = Field(..., description="URL to complete the purchase")
    session_id: str = Field(..., description="Stripe checkout session ID")
    expires_at: datetime = Field(..., description="When the checkout session expires")


class UserPurchase(BaseModel):
    """User purchase record"""

    id: str = Field(..., description="Purchase ID")
    pack_id: PackId = Field(..., description="Credit pack purchased")
    credits_purchased: int = Field(..., description="Number of credits purchased")
    amount_paid: float = Field(..., description="Amount paid in cents")
    currency: str = Field(..., description="Currency code")
    status: str = Field(..., description="Purchase status")
    purchase_date: datetime = Field(..., description="When purchase was made")
    invoice_id: Optional[str] = Field(None, description="Invoice ID")
    stripe_payment_intent_id: Optional[str] = Field(
        None, description="Stripe payment intent ID"
    )


class DownloadInvoiceResponse(BaseModel):
    """Response for invoice download"""

    invoice_url: str = Field(..., description="URL to download the invoice")
    expires_at: datetime = Field(..., description="When the download URL expires")
    format: str = Field(..., description="Invoice format")


class CreditPack(BaseModel):
    """Credit pack information"""

    id: PackId = Field(..., description="Pack ID")
    name: str = Field(..., description="Pack name")
    credits: int = Field(..., description="Number of credits in pack")
    price: float = Field(..., description="Price in cents")
    currency: str = Field(..., description="Currency code")
    popular: bool = Field(False, description="Whether this is the popular choice")
    savings_percentage: Optional[int] = Field(None, description="Savings percentage")


class CreditPacksResponse(BaseModel):
    """Response for available credit packs"""

    packs: List[CreditPack] = Field(..., description="Available credit packs")
    current_credits: int = Field(..., description="User's current credits")


class BillingHealthResponse(BaseModel):
    """Billing system health response"""

    status: str = Field(..., description="System status")
    stripe_connected: bool = Field(..., description="Whether Stripe is connected")
    last_sync: datetime = Field(..., description="Last sync with payment provider")
    active_customers: int = Field(..., description="Number of active customers")


class UserBillingSummary(BaseModel):
    """User billing summary"""

    user_id: str = Field(..., description="User ID")
    current_credits: int = Field(..., description="Current available credits")
    total_purchased: int = Field(..., description="Total credits ever purchased")
    total_spent: float = Field(..., description="Total amount spent")
    last_purchase_date: Optional[datetime] = Field(
        None, description="Last purchase date"
    )
    billing_setup: bool = Field(..., description="Whether billing is set up")

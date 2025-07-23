"""Billing API request models"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PackId(str, Enum):
    """Credit pack IDs"""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class PurchaseCreditsRequest(BaseModel):
    """Request model for purchasing credits"""

    pack_id: PackId = Field(..., description="ID of the credit pack to purchase")
    return_url: Optional[str] = Field(
        None, description="URL to redirect after successful purchase"
    )
    cancel_url: Optional[str] = Field(
        None, description="URL to redirect after cancelled purchase"
    )


class SetupUserRequest(BaseModel):
    """Request model for setting up user billing"""

    user_id: str = Field(..., description="User ID to setup billing for")
    email: Optional[str] = Field(None, description="User email for billing")
    name: Optional[str] = Field(None, description="User name for billing")


class InvoiceRequest(BaseModel):
    """Request model for invoice operations"""

    purchase_id: str = Field(..., description="Purchase ID for the invoice")
    format: Optional[str] = Field("pdf", description="Invoice format (pdf, html)")

"""Billing Service for handling payments and credit management"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from app.api.v1.request.billing_request import PackId, PurchaseCreditsRequest
from app.api.v1.response.billing_response import (
    AvailableCreditsResponse,
    BillingHealthResponse,
    CreditPack,
    CreditPacksResponse,
    DownloadInvoiceResponse,
    PurchaseCreditsResponse,
    SetupUserResponse,
    UserBillingSummary,
    UserPurchase,
)

logger = logging.getLogger(__name__)


class BillingService:
    """Service for handling billing operations"""

    def __init__(self):
        """Initialize billing service"""
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache

        # Mock user data storage (in production, use database)
        self._user_credits = {}
        self._user_purchases = {}
        self._credit_packs = self._get_credit_packs_config()

    def _get_credit_packs_config(self) -> List[CreditPack]:
        """Get available credit pack configurations"""
        return [
            CreditPack(
                id=PackId.STARTER,
                name="Starter Pack",
                credits=1000,
                price=999,  # $9.99
                currency="usd",
                popular=False,
                savings_percentage=None,
            ),
            CreditPack(
                id=PackId.PROFESSIONAL,
                name="Professional Pack",
                credits=5000,
                price=3999,  # $39.99
                currency="usd",
                popular=True,
                savings_percentage=20,
            ),
            CreditPack(
                id=PackId.ENTERPRISE,
                name="Enterprise Pack",
                credits=15000,
                price=9999,  # $99.99
                currency="usd",
                popular=False,
                savings_percentage=33,
            ),
            CreditPack(
                id=PackId.CUSTOM,
                name="Custom Pack",
                credits=50000,
                price=24999,  # $249.99
                currency="usd",
                popular=False,
                savings_percentage=50,
            ),
        ]

    async def get_available_credits(self, user_id: str) -> AvailableCreditsResponse:
        """Get available credits for a user"""
        try:
            logger.info(f"Getting available credits for user {user_id}")

            # Get user credits from mock storage
            credits = self._user_credits.get(user_id, 100)  # Default 100 credits

            return AvailableCreditsResponse(
                credits=credits,
                last_updated=datetime.now(),
                expires_at=datetime.now() + timedelta(days=365),
            )

        except Exception as e:
            logger.error(f"Error getting available credits: {e}")
            raise

    async def purchase_credits(
        self, user_id: str, request: PurchaseCreditsRequest
    ) -> PurchaseCreditsResponse:
        """Initiate credit purchase"""
        try:
            logger.info(
                f"Initiating credit purchase for user {user_id}, pack {request.pack_id}"
            )

            # Find the credit pack
            pack = next(
                (p for p in self._credit_packs if p.id == request.pack_id), None
            )
            if not pack:
                raise ValueError(f"Invalid pack ID: {request.pack_id}")

            # In production, create Stripe checkout session here
            # For demo, generate mock checkout URL
            session_id = f"cs_{uuid4().hex[:24]}"
            checkout_url = f"https://checkout.stripe.com/pay/{session_id}"

            return PurchaseCreditsResponse(
                checkout_url=checkout_url,
                session_id=session_id,
                expires_at=datetime.now() + timedelta(hours=24),
            )

        except Exception as e:
            logger.error(f"Error purchasing credits: {e}")
            raise

    async def setup_user(
        self, user_id: str, email: Optional[str] = None
    ) -> SetupUserResponse:
        """Setup user for billing"""
        try:
            logger.info(f"Setting up billing for user {user_id}")

            # Initialize user with default credits
            if user_id not in self._user_credits:
                self._user_credits[user_id] = 100  # 100 free credits

            # In production, create Stripe customer here
            customer_id = f"cus_{uuid4().hex[:14]}"

            return SetupUserResponse(
                user_id=user_id,
                credits=self._user_credits[user_id],
                billing_setup=True,
                customer_id=customer_id,
            )

        except Exception as e:
            logger.error(f"Error setting up user: {e}")
            raise

    async def get_user_purchase_history(self, user_id: str) -> List[UserPurchase]:
        """Get user's purchase history"""
        try:
            logger.info(f"Getting purchase history for user {user_id}")

            # Get from mock storage or create sample data
            if user_id not in self._user_purchases:
                self._user_purchases[user_id] = self._generate_sample_purchases(user_id)

            return self._user_purchases[user_id]

        except Exception as e:
            logger.error(f"Error getting purchase history: {e}")
            raise

    async def download_invoice(
        self, user_id: str, purchase_id: str
    ) -> DownloadInvoiceResponse:
        """Generate invoice download URL"""
        try:
            logger.info(f"Generating invoice download for purchase {purchase_id}")

            # In production, generate actual invoice from Stripe
            # For demo, create mock download URL
            invoice_url = f"https://invoices.stripe.com/pdf/{purchase_id}"

            return DownloadInvoiceResponse(
                invoice_url=invoice_url,
                expires_at=datetime.now() + timedelta(hours=1),
                format="pdf",
            )

        except Exception as e:
            logger.error(f"Error generating invoice download: {e}")
            raise

    async def get_credit_packs(self, user_id: str) -> CreditPacksResponse:
        """Get available credit packs"""
        try:
            current_credits = await self.get_available_credits(user_id)

            return CreditPacksResponse(
                packs=self._credit_packs, current_credits=current_credits.credits
            )

        except Exception as e:
            logger.error(f"Error getting credit packs: {e}")
            raise

    async def get_billing_health(self) -> BillingHealthResponse:
        """Get billing system health"""
        try:
            return BillingHealthResponse(
                status="healthy",
                stripe_connected=True,  # Mock value
                last_sync=datetime.now() - timedelta(minutes=2),
                active_customers=1250,  # Mock value
            )

        except Exception as e:
            logger.error(f"Error getting billing health: {e}")
            raise

    async def get_user_billing_summary(self, user_id: str) -> UserBillingSummary:
        """Get user billing summary"""
        try:
            credits = await self.get_available_credits(user_id)
            purchases = await self.get_user_purchase_history(user_id)

            total_purchased = sum(p.credits_purchased for p in purchases)
            total_spent = sum(p.amount_paid for p in purchases)
            last_purchase = (
                max([p.purchase_date for p in purchases], default=None)
                if purchases
                else None
            )

            return UserBillingSummary(
                user_id=user_id,
                current_credits=credits.credits,
                total_purchased=total_purchased,
                total_spent=total_spent,
                last_purchase_date=last_purchase,
                billing_setup=True,
            )

        except Exception as e:
            logger.error(f"Error getting billing summary: {e}")
            raise

    def _generate_sample_purchases(self, user_id: str) -> List[UserPurchase]:
        """Generate sample purchase history for demo"""
        import random

        purchases = []

        # Generate 2-5 random purchases
        num_purchases = random.randint(2, 5)

        for i in range(num_purchases):
            pack = random.choice(self._credit_packs)
            purchase_date = datetime.now() - timedelta(days=random.randint(1, 365))

            purchase = UserPurchase(
                id=f"pi_{uuid4().hex[:24]}",
                pack_id=pack.id,
                credits_purchased=pack.credits,
                amount_paid=pack.price,
                currency=pack.currency,
                status="succeeded",
                purchase_date=purchase_date,
                invoice_id=f"in_{uuid4().hex[:24]}",
                stripe_payment_intent_id=f"pi_{uuid4().hex[:24]}",
            )
            purchases.append(purchase)

        # Sort by purchase date (newest first)
        purchases.sort(key=lambda x: x.purchase_date, reverse=True)

        return purchases

    async def add_credits(self, user_id: str, credits: int) -> int:
        """Add credits to user account (internal method for webhook processing)"""
        current_credits = self._user_credits.get(user_id, 0)
        new_credits = current_credits + credits
        self._user_credits[user_id] = new_credits

        logger.info(
            f"Added {credits} credits to user {user_id}. New total: {new_credits}"
        )

        return new_credits

    async def deduct_credits(self, user_id: str, credits: int) -> int:
        """Deduct credits from user account"""
        current_credits = self._user_credits.get(user_id, 0)

        if current_credits < credits:
            raise ValueError(
                f"Insufficient credits. Available: {current_credits}, Required: {credits}"
            )

        new_credits = current_credits - credits
        self._user_credits[user_id] = new_credits

        logger.info(
            f"Deducted {credits} credits from user {user_id}. New total: {new_credits}"
        )

        return new_credits

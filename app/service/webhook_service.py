import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.common.auth.auth import UserContext
from app.common.enum.user import ClerkEvent, UserRole
from app.model.user_model import User

logger = logging.getLogger(__name__)


class WebhookService:
    def __init__(self, db: Session, context: UserContext):
        self.db = db
        self.context = context

    def handle_event(self, event_type: str, data: dict):
        if event_type == ClerkEvent.USER_CREATED:
            self.handle_user_created(data)
        elif event_type == ClerkEvent.USER_UPDATED:
            self.handle_user_updated(data)
        elif event_type == ClerkEvent.USER_DELETED:
            self.handle_user_deleted(data)
        elif event_type == ClerkEvent.ORGANIZATION_CREATED:
            self.handle_organization_created(data)
        elif event_type == ClerkEvent.ORGANIZATION_UPDATED:
            self.handle_organization_updated(data)
        elif event_type == ClerkEvent.ORGANIZATION_DELETED:
            self.handle_organization_deleted(data)
        elif event_type == ClerkEvent.ORGANIZATION_MEMBERSHIP_CREATED:
            self.handle_organization_membership_created(data)
        else:
            logger.warning("Unhandled Clerk event type: %s", event_type)

    def handle_user_created(self, data: dict):
        try:
            # Check if user already exists
            existing_user = (
                self.db.query(User).filter_by(clerk_user_id=data["id"]).first()
            )
            if existing_user:
                logger.info("User already exists: %s", data["id"])
                return  # Skip creation

            # Safely extract optional fields
            phone_number = None
            if data.get("phone_numbers"):
                phone_number = data["phone_numbers"][0].get("phone_number")

            email_address = ""
            if data.get("email_addresses"):
                email_address = data["email_addresses"][0].get("email_address", "")

            # Create user
            user = User(
                clerk_user_id=data["id"],
                name=f"{data.get('first_name', '')} {data.get('last_name', '')}".strip(),
                email=email_address,
                password="",
                role=UserRole.USER,
                phone=phone_number,
                email_verified=data.get("email_verified_at"),
                avatar=data.get("image_url", ""),
            )

            self.db.add(user)
            self.db.commit()
            logger.info("User created: %s", user.clerk_user_id)

        except IntegrityError as e:
            self.db.rollback()
            logger.error("Integrity error creating user: %s", e)
        except Exception as e:
            self.db.rollback()
            logger.error("Error creating user: %s", e)

    def handle_user_updated(self, data: dict):
        try:
            user = self.db.query(User).filter(User.clerk_user_id == data["id"]).first()
            if not user:
                logger.warning("User not found for update: %s", data["id"])
                return

            # Safe phone and email access
            phone_number = None
            if data.get("phone_numbers"):
                phone_number = data["phone_numbers"][0].get("phone_number")

            email_address = ""
            if data.get("email_addresses"):
                email_address = data["email_addresses"][0].get("email_address", "")

            user.name = (
                f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
            )
            user.email = email_address
            user.phone = phone_number
            user.email_verified = data.get("email_verified_at")
            user.avatar = data.get("image_url", "")

            self.db.commit()
            logger.info("User updated: %s", user.clerk_user_id)

        except Exception as e:
            self.db.rollback()
            logger.error("Error updating user: %s", e)

    def handle_user_deleted(self, data: dict):
        try:
            user = self.db.query(User).filter(User.clerk_user_id == data["id"]).first()
            if not user:
                logger.warning("User not found for deletion: %s", data["id"])
                return

            self.db.delete(user)
            self.db.commit()
            logger.info("User deleted: %s", user.clerk_user_id)

        except Exception as e:
            self.db.rollback()
            logger.error("Error deleting user: %s", e)

    # def handle_organization_created(self, data: dict):
    #     try:
    #         organization = Organization(
    #             id=data["id"],
    #             name=data["name"],
    #             slug=data.get("slug"),
    #             logo_url=data.get("image_url"),
    #             public_metadata=data.get("public_metadata"),
    #             private_metadata=data.get("private_metadata")
    #         )
    #         self.db.add(organization)
    #         self.db.commit()
    #         logger.info("Organization created: %s", organization.id)
    #     except IntegrityError as e:
    #         self.db.rollback()
    #         logger.error("Integrity error creating organization: %s", e)
    #     except Exception as e:
    #         self.db.rollback()
    #         logger.error("Error creating organization: %s", e)
    #
    # def handle_organization_updated(self, data: dict):
    #     try:
    #         organization = self.db.query(Organization).filter(Organization.id == data["id"]).first()
    #         if not organization:
    #             logger.warning("Organization not found: %s", data["id"])
    #             return
    #
    #         organization.name = data["name"]
    #         organization.slug = data.get("slug")
    #         organization.logo_url = data.get("image_url")
    #         organization.public_metadata = data.get("public_metadata")
    #         organization.private_metadata = data.get("private_metadata")
    #
    #         self.db.commit()
    #         logger.info("Organization updated: %s", organization.id)
    #     except Exception as e:
    #         self.db.rollback()
    #         logger.error("Error updating organization: %s", e)
    #
    # def handle_organization_deleted(self, data: dict):
    #     try:
    #         organization = self.db.query(Organization).filter(Organization.id == data["id"]).first()
    #         if not organization:
    #             logger.warning("Organization not found: %s", data["id"])
    #             return
    #
    #         self.db.delete(organization)
    #         self.db.commit()
    #         logger.info("Organization deleted: %s", organization.id)
    #     except Exception as e:
    #         self.db.rollback()
    #         logger.error("Error deleting organization: %s", e)
    #
    # def handle_organization_membership_created(self, data: dict):
    #     try:
    #         user = self.db.query(User).filter(User.id == data["user_id"]).first()
    #         if not user:
    #             logger.warning("User not found: %s", data["user_id"])
    #             return
    #
    #         user.organization_id = data["organization"]["id"]
    #         self.db.commit()
    #         logger.info("User %s added to organization %s", user.id, user.organization_id)
    #     except Exception as e:
    #         self.db.rollback()
    #         logger.error("Error adding user to organization: %s", e)

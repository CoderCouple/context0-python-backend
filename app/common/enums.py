from enum import Enum


class WorkflowStatus(str, Enum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"


class EdgeField(str, Enum):
    SOURCE = "source"
    TARGET = "target"


class ExecutionTrigger(str, Enum):
    MANUAL = "MANUAL"
    CRON = "CRON"
    API = "API"


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExecutionPhaseStatus(str, Enum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class UserRole(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"


class ClerkEvent(str, Enum):
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    ORGANIZATION_CREATED = "organization.created"
    ORGANIZATION_UPDATED = "organization.updated"
    ORGANIZATION_DELETED = "organization.deleted"
    ORGANIZATION_MEMBERSHIP_CREATED = "organizationMembership.created"

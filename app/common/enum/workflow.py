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

from enum import Enum


class WorkflowStatus(str, Enum):
    DRAFT = "Draft"
    PUBLISHED = "Published"

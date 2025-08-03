"""Chat-related enums"""
from enum import Enum


class ChatRole(str, Enum):
    """Chat message roles"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSessionStatus(str, Enum):
    """Chat session status"""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

from enum import Enum


class Tags(Enum):
    # Health Check
    Ping = "Ping"

    # Memory
    Memory = "Memory"

    # Q&A and Reasoning
    QA = "Q&A"

    # Workflow
    Workflow = "Workflow"
    Workflow_Execution = "Workflow_Execution"

    # Auth
    Auth = "Auth"
    Webhook = "Webhook"

    # Common
    Permission = "Permission"
    Analytics = "Analytics"
    Credential = "Credential"

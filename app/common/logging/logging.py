import json
import logging
from datetime import datetime

# Create module-level logger
logger = logging.getLogger("contextzero.audit")


def setup_logging():
    """Configures global and SQLAlchemy logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # SQLAlchemy query logging
    sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
    sqlalchemy_logger.setLevel(logging.INFO)

    logger.info("Logging initialized.")

    # Return a logger for the main application
    return logging.getLogger("contextzero.main")


async def log_memory_operation(
    action: str, memory_id: str, user_id: str, operation: str
):
    """Background task for audit logging of memory operations."""
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "memory_id": memory_id,
            "user_id": user_id,
            "operation": operation,
        }
        logger.info(f"Audit log: {json.dumps(log_entry)}")

        # TODO: Write to persistent audit store (e.g., DB, file, or message queue)

    except Exception as e:
        logger.exception(f"Audit logging error: {e}")

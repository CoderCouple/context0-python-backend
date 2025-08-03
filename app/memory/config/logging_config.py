import logging
import os
from typing import Optional

# Memory system logger
memory_logger = logging.getLogger("contextzero.memory")


class MemoryLoggerConfig:
    """Configuration for memory system logging with verbose mode control"""

    def __init__(self):
        self.verbose_config = (
            os.getenv("MEMORY_VERBOSE_CONFIG", "true").lower() == "true"
        )
        self.verbose_operations = (
            os.getenv("MEMORY_VERBOSE_OPERATIONS", "true").lower() == "true"
        )
        self.log_level = os.getenv("MEMORY_LOG_LEVEL", "INFO").upper()

    def setup(self):
        """Setup memory logger with configured verbosity"""
        level = getattr(logging, self.log_level, logging.INFO)
        memory_logger.setLevel(level)

        # Prevent propagation to root logger to avoid duplicate messages
        memory_logger.propagate = False

        # Clear any existing handlers to avoid duplicates
        memory_logger.handlers.clear()

        # Add a single handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        memory_logger.addHandler(handler)

    def log_config(self, message: str, config_data: Optional[dict] = None):
        """Log configuration information if verbose config is enabled"""
        if self.verbose_config:
            if config_data:
                memory_logger.info(f"🔧 {message}: {config_data}")
            else:
                memory_logger.info(f"🔧 {message}")
        else:
            memory_logger.debug(f"{message}: {config_data}" if config_data else message)

    def log_success(self, message: str):
        """Log success messages"""
        if self.verbose_operations:
            memory_logger.info(f"✅ {message}")
        else:
            memory_logger.debug(message)

    def log_error(self, message: str):
        """Log error messages (always shown)"""
        memory_logger.error(f"❌ {message}")

    def log_warning(self, message: str):
        """Log warning messages"""
        memory_logger.warning(f"⚠️ {message}")

    def log_info(self, message: str):
        """Log info messages"""
        if self.verbose_operations:
            memory_logger.info(f"📊 {message}")
        else:
            memory_logger.debug(message)


# Global instance
memory_log_config = MemoryLoggerConfig()

from dataclasses import dataclass

from app.common.enum.system import LogLevel


@dataclass
class SystemConfig:
    """System-level configuration"""

    log_level: LogLevel = LogLevel.INFO
    metrics_enabled: bool = True
    health_check_interval: str = "30s"
    graceful_shutdown_timeout: str = "60s"
    request_timeout: str = "30s"

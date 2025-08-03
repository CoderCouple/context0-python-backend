"""YAML Configuration Loader for Memory System"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from app.memory.config.logging_config import memory_log_config

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv

    # Load environment file based on APP_ENV
    root_path = Path(__file__).resolve().parent.parent.parent.parent
    app_env = os.getenv("APP_ENV", "development")

    # Use .env.dev for development, .env.prod for production
    if app_env == "production":
        env_path = root_path / ".env.prod"
    else:
        env_path = root_path / ".env.dev"

    # Fallback to .env if specific file doesn't exist
    if not env_path.exists():
        env_path = root_path / ".env"

    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # dotenv not available, continue without it
    pass


class YAMLConfigLoader:
    """Load configuration from YAML files with environment variable substitution"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with config file path"""
        if config_path is None:
            # Default to the config directory
            config_dir = Path(__file__).parent
            config_path = config_dir / "default.yaml"

        self.config_path = Path(config_path)
        self._config_data: Optional[Dict[str, Any]] = None

    def load_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration with environment variable substitution"""
        if self._config_data is None:
            self._config_data = self._load_and_substitute()
        return self._config_data

    def _load_and_substitute(self) -> Dict[str, Any]:
        """Load YAML file and substitute environment variables"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        memory_log_config.log_config(f"Loading configuration from: {self.config_path}")

        with open(self.config_path, "r") as file:
            raw_content = file.read()

        # Substitute environment variables
        substituted_content = self._substitute_env_vars(raw_content)

        # Parse YAML
        config = yaml.safe_load(substituted_content)

        return config

    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in YAML content"""
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r"\$\{([^}]+)\}"

        def replace_env_var(match):
            var_expr = match.group(1)

            # Check if it has a default value
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
            else:
                var_name = var_expr
                default_value = None

            # Get environment variable
            env_value = os.getenv(var_name)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # Keep the original placeholder if no value found
                return match.group(0)

        return re.sub(pattern, replace_env_var, content)

    def get_store_config(self, store_type: str) -> Dict[str, Any]:
        """Get configuration for a specific store type"""
        config = self.load_config()
        return config.get(store_type, {})

    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        return self.get_store_config("vector_store")

    def get_graph_store_config(self) -> Dict[str, Any]:
        """Get graph store configuration"""
        return self.get_store_config("graph_store")

    def get_doc_store_config(self) -> Dict[str, Any]:
        """Get document store configuration"""
        return self.get_store_config("doc_store")

    def get_time_store_config(self) -> Dict[str, Any]:
        """Get time series store configuration"""
        return self.get_store_config("time_store")

    def get_audit_store_config(self) -> Dict[str, Any]:
        """Get audit store configuration"""
        return self.get_store_config("audit_store")

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        config = self.load_config()
        return config.get("llm", {})

    def get_embedder_config(self) -> Dict[str, Any]:
        """Get embedder configuration"""
        config = self.load_config()
        return config.get("embedder", {})

    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        config = self.load_config()
        return config.get("system", {})


# Global configuration loader instance
_config_loader = None


def get_config_loader() -> YAMLConfigLoader:
    """Get global configuration loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = YAMLConfigLoader()
    return _config_loader

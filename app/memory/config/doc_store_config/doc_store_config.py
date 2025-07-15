import importlib
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator

from app.memory.config.base import BaseStoreConfig


class DocumentStoreConfig(BaseModel):
    """Dynamic document store configuration supporting multiple providers"""

    provider: str = Field(default="mongodb", description="Document store provider")

    config: Optional[Union[Dict[str, Any], BaseStoreConfig]] = Field(
        default=None, description="Provider-specific configuration"
    )

    # Registry of supported providers and their config classes
    _provider_configs: Dict[str, str] = {
        "mongodb": "MongoDBConfig",
        "elasticsearch": "ElasticsearchConfig",
        "in_memory": "InMemoryDocConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "DocumentStoreConfig":
        """Dynamically load and validate provider-specific configuration"""
        provider = self.provider.lower()
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(
                f"Unsupported document store provider: {provider}. "
                f"Supported providers: {list(self._provider_configs.keys())}"
            )

        try:
            # Dynamic import of provider-specific config
            module = importlib.import_module(
                f"app.memory.config.doc_store_config.{provider}_config"
            )
            config_class = getattr(module, self._provider_configs[provider])
        except ImportError as e:
            raise ValueError(
                f"Provider '{provider}' configuration module not found. "
                f"Please install the required dependencies or check the module path. Error: {e}"
            )
        except AttributeError as e:
            raise ValueError(
                f"Configuration class '{self._provider_configs[provider]}' "
                f"not found in provider module. Error: {e}"
            )

        # Handle different config input types
        if config is None:
            config = {}

        if isinstance(config, config_class):
            # Already the right type
            return self

        if not isinstance(config, dict):
            raise ValueError(
                f"Config must be a dict or {config_class.__name__} instance, "
                f"got {type(config)}"
            )

        # Create the provider-specific config instance
        try:
            self.config = config_class(**config)
        except Exception as e:
            raise ValueError(f"Failed to create {provider} configuration: {e}")

        return self

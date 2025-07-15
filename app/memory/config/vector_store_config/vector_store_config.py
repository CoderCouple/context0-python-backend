import importlib
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator

from app.common.enum.persistence import PersistenceType, VectorStoreProvider
from app.memory.config.base import BaseStoreConfig


class VectorStoreConfig(BaseModel):
    """Dynamic vector store configuration supporting multiple providers"""

    provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.PINECONE, description="Vector store provider"
    )

    config: Optional[Union[Dict[str, Any], BaseStoreConfig]] = Field(
        default=None, description="Provider-specific configuration"
    )

    # Registry of supported providers and their config classes
    _provider_configs: Dict[str, str] = {
        "in_memory": "InMemoryVectorConfig",
        "qdrant": "QdrantConfig",
        "weaviate": "WeaviateConfig",
        "faiss": "FAISSConfig",
        "milvus": "MilvusConfig",
        "pinecone": "PineconeConfig",
        "chroma": "ChromaConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "VectorStoreConfig":
        """Dynamically load and validate provider-specific configuration"""
        provider = self.provider
        config = self.config

        # Convert enum to string value
        provider_name = (
            provider.value.lower()
            if hasattr(provider, "value")
            else str(provider).lower()
        )

        if provider_name not in self._provider_configs:
            raise ValueError(
                f"Unsupported vector store provider: {provider_name}. "
                f"Supported providers: {list(self._provider_configs.keys())}"
            )

        try:
            # Dynamic import of provider-specific config
            module = importlib.import_module(
                f"app.memory.config.vector_store_config.{provider_name}_config"
            )
            config_class = getattr(module, self._provider_configs[provider_name])
        except ImportError as e:
            raise ValueError(
                f"Provider '{provider_name}' configuration module not found. "
                f"Please install the required dependencies or check the module path. Error: {e}"
            )
        except AttributeError as e:
            raise ValueError(
                f"Configuration class '{self._provider_configs[provider_name]}' "
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

        # Auto-set provider-specific defaults
        config = self._apply_provider_defaults(provider_name, config, config_class)

        # Create the provider-specific config instance
        try:
            self.config = config_class(**config)
        except Exception as e:
            raise ValueError(f"Failed to create {provider_name} configuration: {e}")

        return self

    def _apply_provider_defaults(
        self, provider: str, config: Dict[str, Any], config_class: Type[BaseStoreConfig]
    ) -> Dict[str, Any]:
        """Apply intelligent defaults based on provider and config class"""

        # File-based providers get automatic path setup
        if "path" not in config and "path" in config_class.__annotations__:
            config["path"] = f"/tmp/memory_ai/{provider}"

        # Local providers get memory persistence by default in development
        if provider in ["in_memory", "faiss"] and "persistence" not in config:
            config["persistence"] = PersistenceType.MEMORY.value

        # Set collection/index name based on provider conventions
        if (
            "collection_name" not in config
            and "collection_name" in config_class.__annotations__
        ):
            config["collection_name"] = "memories"
        elif (
            "index_name" not in config and "index_name" in config_class.__annotations__
        ):
            config["index_name"] = "memories"

        # Vector size defaults
        if (
            "vector_size" not in config
            and "vector_size" in config_class.__annotations__
        ):
            config["vector_size"] = 1536  # Default for OpenAI embeddings

        return config

    @classmethod
    def register_provider(cls, name: str, config_class_name: str):
        """Register a new provider at runtime (for plugins)"""
        cls._provider_configs[name] = config_class_name

    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """Get list of available providers"""
        return cls._provider_configs.copy()

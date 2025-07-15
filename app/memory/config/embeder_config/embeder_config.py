import importlib
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, model_validator


class EmbedderConfig(BaseModel):
    """Dynamic embedder configuration"""

    provider: str = Field(
        default="openai",
        description="Embedding provider (e.g., 'openai', 'huggingface', 'sentence-transformers')",
    )
    config: Optional[Union[Dict[str, Any], BaseModel]] = Field(
        default=None, description="Provider-specific configuration"
    )

    _provider_configs: Dict[str, str] = {
        "openai": "OpenAIEmbedderConfig",
        "huggingface": "HuggingFaceEmbedderConfig",
        "sentence_transformers": "SentenceTransformersConfig",
        "cohere": "CohereEmbedderConfig",
        "azure": "AzureEmbedderConfig",
        "vertex_ai": "VertexAIEmbedderConfig",
        "local": "LocalEmbedderConfig",
        "ollama": "OllamaEmbedderConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "EmbedderConfig":
        """Dynamically load and validate embedder configuration"""
        provider = self.provider
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported embedder provider: {provider}")

        try:
            module = importlib.import_module(
                f"app.memory.config.embeder_config.{provider}_config"
            )
            config_class = getattr(module, self._provider_configs[provider])
        except ImportError:
            raise ValueError(f"Embedder provider '{provider}' module not found")

        if config is None:
            config = {}

        if isinstance(config, config_class):
            return self

        # Apply embedder-specific defaults
        if "model" not in config and "model" in config_class.__annotations__:
            default_models = {
                "openai": "text-embedding-3-small",
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
                "sentence_transformers": "all-MiniLM-L6-v2",
            }
            config["model"] = default_models.get(provider, "default")

        self.config = config_class(**config)
        return self

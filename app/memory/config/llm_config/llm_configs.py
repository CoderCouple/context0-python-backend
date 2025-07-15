import importlib
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, model_validator


class LlmConfig(BaseModel):
    """Dynamic LLM configuration"""

    provider: str = Field(
        default="openai",
        description="LLM provider (e.g., 'openai', 'anthropic', 'azure')",
    )
    config: Optional[Union[Dict[str, Any], BaseModel]] = Field(
        default=None, description="Provider-specific configuration"
    )

    _provider_configs: Dict[str, str] = {
        "openai": "OpenAIConfig",
        "anthropic": "AnthropicConfig",
        "azure": "AzureOpenAIConfig",
        "huggingface": "HuggingFaceConfig",
        "cohere": "CohereConfig",
        "palm": "PaLMConfig",
        "claude": "ClaudeConfig",
        "local": "LocalLLMConfig",
        "ollama": "OllamaConfig",
        "bedrock": "BedrockConfig",
        "vertex_ai": "VertexAILLMConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "LlmConfig":
        """Dynamically load and validate LLM provider configuration"""
        provider = self.provider
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        try:
            module = importlib.import_module(
                f"app.memory.config.llm_config.{provider}_config"
            )
            config_class = getattr(module, self._provider_configs[provider])
        except ImportError:
            raise ValueError(f"LLM provider '{provider}' module not found")

        if config is None:
            config = {}

        if isinstance(config, config_class):
            return self

        # Apply LLM-specific defaults
        if "model" not in config and "model" in config_class.__annotations__:
            default_models = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-sonnet-20241022",
                "azure": "gpt-4o-mini",
                "local": "llama2",
            }
            config["model"] = default_models.get(provider, "default")

        if (
            "temperature" not in config
            and "temperature" in config_class.__annotations__
        ):
            config["temperature"] = 0.1

        self.config = config_class(**config)
        return self

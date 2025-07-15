import importlib
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, model_validator

from app.memory.config.base import BaseStoreConfig


class GraphStoreConfig(BaseModel):
    """Dynamic graph store configuration"""

    provider: str = Field(
        default="in_memory",
        description="Graph store provider (e.g., 'neo4j', 'memgraph', 'arangodb')",
    )
    config: Optional[Union[Dict[str, Any], BaseStoreConfig]] = Field(
        default=None, description="Provider-specific configuration"
    )

    _provider_configs: Dict[str, str] = {
        "in_memory": "InMemoryGraphConfig",
        "neo4j": "Neo4jConfig",
        "memgraph": "MemgraphConfig",
        "arangodb": "ArangoDBConfig",
        "neptune": "NeptuneConfig",
        "tigergraph": "TigerGraphConfig",
        "dgraph": "DGraphConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "GraphStoreConfig":
        """Dynamically load and validate graph provider configuration"""
        provider = self.provider
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported graph store provider: {provider}")

        try:
            module = importlib.import_module(
                f"app.memory.config.graph_store_config.{provider}_config"
            )
            config_class = getattr(module, self._provider_configs[provider])
        except ImportError:
            raise ValueError(f"Graph provider '{provider}' module not found")

        if config is None:
            config = {}

        if isinstance(config, config_class):
            return self

        if not isinstance(config, dict):
            raise ValueError(
                f"Config must be a dict or {config_class.__name__} instance"
            )

        # Apply graph-specific defaults
        if "database" not in config and "database" in config_class.__annotations__:
            config["database"] = "memory_graph"

        self.config = config_class(**config)
        return self

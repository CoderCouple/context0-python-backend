from typing import Optional

from pydantic import Field

from app.memory.config.base import BaseStoreConfig


class InMemoryGraphConfig(BaseStoreConfig):
    """In-memory graph store configuration"""

    max_nodes: int = Field(default=10000, description="Maximum number of nodes")
    max_edges: int = Field(default=50000, description="Maximum number of edges")

    class Config:
        env_prefix = "GRAPH_MEMORY_"

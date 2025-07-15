from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.memory.config.doc_store_config.doc_store_config import DocumentStoreConfig
from app.memory.config.embeder_config.embeder_config import EmbedderConfig
from app.memory.config.graph_store_config.graph_store_config import GraphStoreConfig
from app.memory.config.llm_config.llm_configs import LlmConfig
from app.memory.config.relational_store_config.relational_store_config import (
    RelationalStoreConfig,
)
from app.memory.config.vector_store_config.vector_store_config import VectorStoreConfig


class ContextZeroConfig(BaseModel):
    """Dynamic ContextZero configuration with extensible provider support"""

    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph store",
        default_factory=GraphStoreConfig,
    )
    document_store: DocumentStoreConfig = Field(
        description="Configuration for the document store",
        default_factory=DocumentStoreConfig,
    )
    relational_store: RelationalStoreConfig = Field(
        description="Configuration for the relational store",
        default_factory=RelationalStoreConfig,
    )
    audit_store: DocumentStoreConfig = Field(
        description="Configuration for the audit store (uses document store)",
        default_factory=DocumentStoreConfig,
    )

    # System configuration
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    environment: str = Field(
        description="Environment (development, staging, production)",
        default="development",
    )

    # Custom prompts (your feature)
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update memory",
        default=None,
    )
    custom_consolidation_prompt: Optional[str] = Field(
        description="Custom prompt for memory consolidation",
        default=None,
    )

    # Feature flags
    enable_graph_memory: bool = Field(
        description="Enable graph-based memory relationships", default=True
    )
    enable_memory_consolidation: bool = Field(
        description="Enable automatic memory consolidation", default=True
    )
    enable_privacy_mode: bool = Field(
        description="Enable privacy protection features", default=True
    )

    class Config:
        env_prefix = "CONTEXT_ZERO_"
        extra = "allow"  # Allow additional fields for future extensibility

    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured providers"""
        return {
            "vector_store": {
                "provider": self.vector_store.provider,
                "config_type": type(self.vector_store.config).__name__
                if self.vector_store.config
                else None,
                "available_providers": self.vector_store.get_available_providers(),
            },
            "llm": {
                "provider": self.llm.provider,
                "config_type": type(self.llm.config).__name__
                if self.llm.config
                else None,
                "available_providers": self.llm._provider_configs,
            },
            "embedder": {
                "provider": self.embedder.provider,
                "config_type": type(self.embedder.config).__name__
                if self.embedder.config
                else None,
                "available_providers": self.embedder._provider_configs,
            },
            "graph_store": {
                "provider": self.graph_store.provider,
                "config_type": type(self.graph_store.config).__name__
                if self.graph_store.config
                else None,
                "available_providers": self.graph_store._provider_configs,
            },
            "document_store": {
                "provider": self.document_store.provider,
                "config_type": type(self.document_store.config).__name__
                if self.document_store.config
                else None,
                "available_providers": self.document_store._provider_configs,
            },
            "relational_store": {
                "provider": self.relational_store.provider,
                "config_type": type(self.relational_store.config).__name__
                if self.relational_store.config
                else None,
                "available_providers": self.relational_store._provider_configs,
            },
            "audit_store": {
                "provider": self.audit_store.provider,
                "config_type": type(self.audit_store.config).__name__
                if self.audit_store.config
                else None,
                "available_providers": self.audit_store._provider_configs,
            },
        }

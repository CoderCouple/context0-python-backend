# ===== CONFIGURATION FACTORY =====
from app.memory.config.context_zero_config import ContextZeroConfig
from app.memory.config.embeder_config.embeder_config import EmbedderConfig
from app.memory.config.graph_store_config.graph_store_config import GraphStoreConfig
from app.memory.config.llm_config.llm_configs import LlmConfig
from app.memory.config.vector_store_config.vector_store_config import VectorStoreConfig


class ConfigFactory:
    """Factory for creating configurations with presets"""

    @staticmethod
    def create_development_config() -> ContextZeroConfig:
        """Create development configuration with available stores"""
        return ContextZeroConfig(
            environment="development",
            vector_store=VectorStoreConfig(
                provider="pinecone",
                config={
                    "api_key": "${PINECONE_API_KEY}",
                    "environment": "${PINECONE_ENVIRONMENT}",
                    "index_name": "${PINECONE_INDEX_NAME}",
                },
            ),
            graph_store=GraphStoreConfig(
                provider="in_memory", config={"max_nodes": 10000}
            ),
            llm=LlmConfig(
                provider="openai", config={"model": "gpt-4o-mini", "temperature": 0.1}
            ),
            embedder=EmbedderConfig(
                provider="openai", config={"model": "text-embedding-3-small"}
            ),
        )

    @staticmethod
    def create_production_config() -> ContextZeroConfig:
        """Create production configuration with persistent stores"""
        return ContextZeroConfig(
            environment="production",
            vector_store=VectorStoreConfig(
                provider="qdrant",
                config={
                    "host": "${QDRANT_HOST}",
                    "port": 6333,
                    "api_key": "${QDRANT_API_KEY}",
                    "collection_name": "memories_prod",
                },
            ),
            graph_store=GraphStoreConfig(
                provider="neo4j",
                config={
                    "host": "${NEO4J_HOST}",
                    "username": "${NEO4J_USERNAME}",
                    "password": "${NEO4J_PASSWORD}",
                },
            ),
            llm=LlmConfig(
                provider="openai",
                config={
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                },
            ),
        )


# ===== USAGE EXAMPLES =====


def example_usage():
    """Example of how to use the dynamic configuration"""

    # Example 1: Simple configuration
    config = ContextZeroConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant", config={"host": "localhost", "port": 6333}
        )
    )

    # Example 2: Using factory presets
    dev_config = ConfigFactory.create_development_config()
    prod_config = ConfigFactory.create_production_config()

    # Example 3: Runtime provider registration (for plugins)
    VectorStoreConfig.register_provider("my_custom_db", "MyCustomDBConfig")

    # Example 4: Configuration from dict (useful for loading from files)
    config_dict = {
        "vector_store": {
            "provider": "pinecone",
            "config": {
                "api_key": "pk-xxx",
                "environment": "us-west1-gcp",
                "index_name": "my-index",
            },
        },
        "llm": {
            "provider": "anthropic",
            "config": {"api_key": "sk-ant-xxx", "model": "claude-3-5-sonnet-20241022"},
        },
    }

    dynamic_config = ContextZeroConfig(**config_dict)

    # Example 5: Get provider information
    provider_info = dynamic_config.get_provider_info()
    print(
        "Available vector store providers:",
        provider_info["vector_store"]["available_providers"],
    )

    return config, dev_config, prod_config, dynamic_config


if __name__ == "__main__":
    example_usage()

#!/usr/bin/env python3
"""Detailed test script to debug memory engine configuration"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_environment_variables():
    """Test that environment variables are properly loaded"""
    print("Testing environment variables...")

    env_vars = [
        "PINECONE_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "MONGODB_CONNECTION_STRING",
        "TIMESCALE_CONNECTION_STRING",
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(
                f"✅ {var}: {value[:20]}..." if len(value) > 20 else f"✅ {var}: {value}"
            )
        else:
            print(f"❌ {var}: Not set")


async def test_config_substitution():
    """Test that YAML config properly substitutes environment variables"""
    print("\nTesting YAML config substitution...")

    try:
        from app.memory.config.yaml_config_loader import get_config_loader

        config_loader = get_config_loader()

        # Test each store config individually
        stores = {
            "vector_store": config_loader.get_vector_store_config(),
            "graph_store": config_loader.get_graph_store_config(),
            "doc_store": config_loader.get_doc_store_config(),
            "time_store": config_loader.get_time_store_config(),
            "audit_store": config_loader.get_audit_store_config(),
        }

        for store_name, store_config in stores.items():
            print(f"\n{store_name}:")
            print(f"  Provider: {store_config.get('provider')}")
            config_dict = store_config.get("config", {})
            for key, value in config_dict.items():
                # Check if environment variable substitution worked
                if isinstance(value, str) and value.startswith("${"):
                    print(f"  ❌ {key}: {value} (substitution failed)")
                else:
                    display_value = (
                        value[:30] + "..."
                        if isinstance(value, str) and len(value) > 30
                        else value
                    )
                    print(f"  ✅ {key}: {display_value}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_individual_store_initialization():
    """Test each store type individually"""
    print("\nTesting individual store initialization...")

    try:
        from app.memory.config.yaml_config_loader import get_config_loader

        config_loader = get_config_loader()

        # Test Pinecone
        print("\n--- Testing Pinecone ---")
        try:
            vector_config = config_loader.get_vector_store_config()
            store_config = vector_config.get("config", {})
            print(f"API Key present: {bool(store_config.get('api_key'))}")
            print(f"Environment: {store_config.get('environment')}")
            print(f"Index name: {store_config.get('index_name')}")

            from app.memory.stores.pinecone_store import PineconeVectorStore

            pinecone_store = PineconeVectorStore(
                api_key=store_config.get("api_key"),
                environment=store_config.get("environment", "us-east-1-aws"),
                index_name=store_config.get("index_name", "memory-index"),
            )
            await pinecone_store.initialize()
            print("✅ Pinecone initialization successful")
        except Exception as e:
            print(f"❌ Pinecone error: {e}")

        # Test Neo4j
        print("\n--- Testing Neo4j ---")
        try:
            graph_config = config_loader.get_graph_store_config()
            store_config = graph_config.get("config", {})
            print(f"URI: {store_config.get('uri')}")
            print(f"Username: {store_config.get('username')}")
            print(f"Password present: {bool(store_config.get('password'))}")

            from app.memory.stores.neo4j_store import Neo4jGraphStore

            neo4j_store = Neo4jGraphStore(
                uri=store_config.get("uri"),
                username=store_config.get("username"),
                password=store_config.get("password"),
                database=store_config.get("database", "neo4j"),
            )
            await neo4j_store.initialize()
            print("✅ Neo4j initialization successful")
        except Exception as e:
            print(f"❌ Neo4j error: {e}")

        # Test MongoDB
        print("\n--- Testing MongoDB ---")
        try:
            doc_config = config_loader.get_doc_store_config()
            store_config = doc_config.get("config", {})
            connection_string = store_config.get("connection_string")
            print(f"Connection string present: {bool(connection_string)}")
            if connection_string:
                print(f"Connection string starts with: {connection_string[:30]}...")

            from app.memory.stores.mongodb_store import MongoDocumentStore

            mongo_store = MongoDocumentStore(
                connection_string=connection_string,
                database_name=store_config.get("database_name", "memory_system"),
                collection_name=store_config.get("collection_name", "memories"),
            )
            await mongo_store.initialize()
            print("✅ MongoDB initialization successful")
        except Exception as e:
            print(f"❌ MongoDB error: {e}")

        # Test TimescaleDB
        print("\n--- Testing TimescaleDB ---")
        try:
            time_config = config_loader.get_time_store_config()
            store_config = time_config.get("config", {})
            connection_string = store_config.get("connection_string")
            print(f"Connection string present: {bool(connection_string)}")
            if connection_string:
                print(f"Connection string starts with: {connection_string[:30]}...")

            from app.memory.stores.timescale_store import TimescaleTimeSeriesStore

            timescale_store = TimescaleTimeSeriesStore(
                connection_string=connection_string,
                table_name=store_config.get("table_name", "memory_timeseries"),
            )
            await timescale_store.initialize()
            print("✅ TimescaleDB initialization successful")
        except Exception as e:
            print(f"❌ TimescaleDB error: {e}")

    except Exception as e:
        print(f"❌ Overall error: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests"""
    await test_environment_variables()
    await test_config_substitution()
    await test_individual_store_initialization()


if __name__ == "__main__":
    asyncio.run(main())

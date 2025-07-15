#!/usr/bin/env python3
"""Test script to verify memory engine configuration loading"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_config_loading():
    """Test the YAML configuration loading"""
    print("Testing YAML configuration loading...")

    try:
        from app.memory.config.yaml_config_loader import get_config_loader

        config_loader = get_config_loader()
        print("✅ Config loader created successfully")

        # Test loading configuration
        config = config_loader.load_config()
        print("✅ Configuration loaded successfully")
        print(f"Config keys: {list(config.keys())}")

        # Test individual store configs
        vector_config = config_loader.get_vector_store_config()
        print(f"✅ Vector store config: {vector_config}")

        graph_config = config_loader.get_graph_store_config()
        print(f"✅ Graph store config: {graph_config}")

        doc_config = config_loader.get_doc_store_config()
        print(f"✅ Doc store config: {doc_config}")

        time_config = config_loader.get_time_store_config()
        print(f"✅ Time store config: {time_config}")

        audit_config = config_loader.get_audit_store_config()
        print(f"✅ Audit store config: {audit_config}")

    except Exception as e:
        print(f"❌ Error testing config loading: {e}")
        import traceback

        traceback.print_exc()


async def test_memory_engine():
    """Test memory engine initialization"""
    print("\nTesting memory engine initialization...")

    try:
        from app.memory.engine.memory_engine import MemoryEngine

        engine = MemoryEngine.get_instance()
        print("✅ Memory engine instance created")

        await engine.initialize()
        print("✅ Memory engine initialized")

        # Check what stores were initialized
        stores_status = {
            "vector_store": engine.vector_store is not None,
            "graph_store": engine.graph_store is not None,
            "doc_store": engine.doc_store is not None,
            "timeseries_store": engine.timeseries_store is not None,
            "audit_store": engine.audit_store is not None,
        }

        print(f"✅ Stores status: {stores_status}")

        # Test health check
        health = await engine.health_check()
        print(f"✅ Health check: {health}")

    except Exception as e:
        print(f"❌ Error testing memory engine: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests"""
    await test_config_loading()
    await test_memory_engine()


if __name__ == "__main__":
    asyncio.run(main())

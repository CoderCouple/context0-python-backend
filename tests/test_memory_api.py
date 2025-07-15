#!/usr/bin/env python3
"""Test memory API functionality"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_memory_service():
    """Test memory service functionality"""
    print("Testing memory service...")

    try:
        # Load environment variables
        from dotenv import load_dotenv

        load_dotenv()

        # Import memory service
        from app.api.v1.request.memory_request import MemoryRecordInput
        from app.service.memory_service import MemoryService

        # Create memory service
        memory_service = MemoryService()
        print("✅ Memory service created")

        # Test system stats (health check)
        stats = await memory_service.get_system_stats()
        print("✅ System stats retrieved")
        print(f"Memory count: {stats.memory_count}")
        print(f"Store status: {json.dumps(stats.stores, indent=2)}")

        # Test adding a simple memory
        memory_input = MemoryRecordInput(
            user_id="test-user-123",
            session_id="test-session-456",
            text="This is a test memory for the memory system",
            tags=["test", "demo"],
            metadata={"source": "test_script"},
        )

        print("\nTesting memory addition...")
        response = await memory_service.create_memory(memory_input)
        print(f"✅ Memory added: {response.success}")
        print(f"Memory ID: {response.memory_id}")
        print(f"Processing time: {response.processing_time_ms}ms")
        print(f"Message: {response.message}")

        if response.success:
            print("\n🎉 Memory system is working correctly!")
            print("Data should now be visible in your cloud databases:")
            print("- Pinecone: Check your memory-index")
            print("- Neo4j Aura: Check for new nodes and relationships")
            print("- MongoDB Atlas: Check memory_system.memories collection")
            print("- TimescaleDB Cloud: Check memory_timeseries table")
        else:
            print(f"\n❌ Memory addition failed: {response.message}")

    except Exception as e:
        print(f"❌ Error testing memory service: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run memory service test"""
    await test_memory_service()


if __name__ == "__main__":
    asyncio.run(main())

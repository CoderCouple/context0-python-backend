#!/usr/bin/env python3
"""Debug memory content retrieval"""

import asyncio
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.service.qa_service import QAService


async def debug_memory_content():
    """Debug what _get_actual_memory_content returns"""
    print("🔍 Debugging memory content retrieval...")

    qa_service = QAService()

    # Initialize memory engine
    await qa_service.memory_engine.initialize()

    # Test with a known memory ID from fresh data
    memory_id = "478f62fa-69b8-4bf7-8b16-0ed7b6625b2e"  # From the API response
    user_id = "john-doe-fresh"

    print(f"Testing memory ID: {memory_id}")
    print(f"User ID: {user_id}")

    # Test the method
    result = await qa_service._get_actual_memory_content(memory_id, user_id)

    if result:
        print(f"✅ Memory content retrieved:")
        print(f"   Memory ID: {result.get('memory_id')}")
        print(f"   Text: {result.get('text', 'NO TEXT')}")
        print(f"   Summary: {result.get('summary', 'NO SUMMARY')}")
        print(f"   Memory Type: {result.get('memory_type')}")
        print(f"   Tags: {result.get('tags', [])}")
    else:
        print(f"❌ No memory content retrieved")

    # Also test vector store directly
    print(f"\n🔍 Testing vector store directly...")
    if qa_service.memory_engine.vector_store:
        from app.common.enum.memory import MemoryType

        handler = qa_service.memory_engine.router.get_handler(
            MemoryType.SEMANTIC_MEMORY
        )
        dummy_embedding, _ = await handler.extract_embedding("test")

        search_filter = {"user_id": user_id, "memory_id": memory_id}

        vector_results = await qa_service.memory_engine.vector_store.similarity_search(
            embedding=dummy_embedding, limit=1, filter=search_filter
        )

        if vector_results:
            memory_entry, score = vector_results[0]
            print(f"✅ Vector store returned: {type(memory_entry)}")
            if isinstance(memory_entry, dict):
                metadata = memory_entry.get("metadata", {})
                print(f"   ID: {memory_entry.get('id')}")
                print(f"   Input: {metadata.get('input', 'NO INPUT')}")
                print(f"   Summary: {metadata.get('summary', 'NO SUMMARY')}")
                print(f"   All metadata keys: {list(metadata.keys())}")
        else:
            print(f"❌ No vector results found")


if __name__ == "__main__":
    asyncio.run(debug_memory_content())

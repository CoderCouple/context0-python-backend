#!/usr/bin/env python3
"""Debug memory search functionality"""

import asyncio
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api.v1.request.memory_request import SearchQuery
from app.memory.engine.memory_engine import MemoryEngine


async def debug_search():
    """Debug the search functionality step by step"""
    print("🔍 Starting Search Debug...")

    try:
        # Initialize engine
        print("1. Initializing memory engine...")
        engine = MemoryEngine.get_instance()
        await engine.initialize()

        print(f"   Vector store: {'✅' if engine.vector_store else '❌'}")
        print(f"   Document store: {'✅' if engine.doc_store else '❌'}")
        print(f"   Graph store: {'✅' if engine.graph_store else '❌'}")
        print(f"   Router: {'✅' if engine.router else '❌'}")

        # Test search query
        print("\n2. Creating search query...")
        query = SearchQuery(
            user_id="john-doe", query="MIT", threshold=0.0, include_content=True
        )
        print(f"   Query: {query.query}")
        print(f"   User: {query.user_id}")

        # Test handler
        print("\n3. Testing handler...")
        from app.common.enum.memory import MemoryType

        handler = engine.router.get_handler(MemoryType.SEMANTIC_MEMORY)
        print(f"   Handler: {'✅' if handler else '❌'}")

        if handler:
            print("   Testing embedding generation...")
            try:
                embedding, summary = await handler.extract_embedding("test query")
                print(
                    f"   Embedding: {'✅' if embedding else '❌'} (length: {len(embedding) if embedding else 0})"
                )
            except Exception as e:
                print(f"   Embedding error: {e}")

        # Test vector store directly
        print("\n4. Testing vector store search...")
        if engine.vector_store:
            try:
                embedding, _ = await handler.extract_embedding(query.query)
                search_filter = {"user_id": query.user_id}

                print(f"   Filter: {search_filter}")
                print(f"   Embedding length: {len(embedding)}")

                vector_results = await engine.vector_store.similarity_search(
                    embedding=embedding, limit=query.limit, filter=search_filter
                )

                print(f"   Vector results: {len(vector_results)}")
                for i, (memory_entry, score) in enumerate(vector_results):
                    if isinstance(memory_entry, dict):
                        entry_id = memory_entry.get("id", "unknown")
                        metadata = memory_entry.get("metadata", {})
                        print(f"   Result {i+1}: {entry_id} (score: {score})")
                        print(f"     Memory type: {metadata.get('memory_type')}")
                        print(f"     Created at: {metadata.get('created_at')}")
                        print(f"     Input: {metadata.get('input', 'NO INPUT FIELD')}")
                        print(f"     Text: {metadata.get('text', 'NO TEXT FIELD')}")
                        print(f"     Summary: {metadata.get('summary', 'NO SUMMARY')}")
                        print(f"     All metadata keys: {list(metadata.keys())}")
                    else:
                        entry_id = memory_entry.id
                        print(f"   Result {i+1}: {entry_id} (score: {score})")

            except Exception as e:
                print(f"   Vector search error: {e}")
                import traceback

                traceback.print_exc()

        # Test search
        print("\n5. Testing full search...")
        response = await engine.search_memories(query)

        print(f"   Success: {response.success}")
        print(f"   Results: {len(response.results)}")
        print(f"   Query time: {response.query_time_ms}ms")

        if response.results:
            for i, result in enumerate(response.results):
                print(f"   Result {i+1}: {result.id} (score: {result.score})")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_search())

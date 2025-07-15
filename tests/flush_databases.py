#!/usr/bin/env python3
"""
Database flush script - Clean all databases for fresh start
Removes all data from Vector, Document, Graph, TimeSeries, and Audit stores
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.memory.config.config_factory import ConfigFactory
from app.memory.engine.memory_engine import MemoryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseFlusher:
    """Utility to flush all databases for clean restart"""

    def __init__(self):
        self.memory_engine = None
        self.flushed_stores = []

    async def initialize(self):
        """Initialize memory engine and stores"""
        try:
            config = ConfigFactory.create_development_config()
            self.memory_engine = MemoryEngine(config)
            await self.memory_engine.initialize()
            logger.info("Memory engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory engine: {e}")
            raise

    async def flush_vector_store(self):
        """Flush all data from vector store (Pinecone)"""
        try:
            logger.info("🗑️  Flushing Vector Store (Pinecone)...")
            vector_store = self.memory_engine.vector_store

            # Get all vector IDs and delete them
            # Note: This is a simplified approach - production might need batch deletion
            await vector_store.delete_all_vectors()

            self.flushed_stores.append("vector_store")
            logger.info("   ✅ Vector store flushed")

        except Exception as e:
            logger.error(f"   ❌ Failed to flush vector store: {e}")

    async def flush_document_store(self):
        """Flush all data from document store (MongoDB)"""
        try:
            logger.info("🗑️  Flushing Document Store (MongoDB)...")
            doc_store = self.memory_engine.doc_store

            # Drop all collections related to memories
            collections_to_drop = [
                "memories",
                "memory_metadata",
                "memory_sessions",
                "memory_chunks",
            ]

            for collection_name in collections_to_drop:
                try:
                    await doc_store.database.drop_collection(collection_name)
                    logger.info(f"   Dropped collection: {collection_name}")
                except Exception as e:
                    logger.warning(
                        f"   Collection {collection_name} may not exist: {e}"
                    )

            self.flushed_stores.append("document_store")
            logger.info("   ✅ Document store flushed")

        except Exception as e:
            logger.error(f"   ❌ Failed to flush document store: {e}")

    async def flush_graph_store(self):
        """Flush all data from graph store (Neo4j)"""
        try:
            logger.info("🗑️  Flushing Graph Store (Neo4j)...")
            graph_store = self.memory_engine.graph_store

            # Delete all nodes and relationships
            queries = [
                "MATCH (n) DETACH DELETE n",  # Delete all nodes and relationships
                "CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *",  # Reset schema if APOC available
            ]

            for query in queries:
                try:
                    await graph_store.execute_query(query)
                    logger.info(f"   Executed: {query[:50]}...")
                except Exception as e:
                    logger.warning(f"   Query failed (may be expected): {e}")

            self.flushed_stores.append("graph_store")
            logger.info("   ✅ Graph store flushed")

        except Exception as e:
            logger.error(f"   ❌ Failed to flush graph store: {e}")

    async def flush_timescale_store(self):
        """Flush all data from TimescaleDB"""
        try:
            logger.info("🗑️  Flushing TimescaleDB...")
            timescale_store = self.memory_engine.timescale_store

            # Drop and recreate tables
            tables_to_drop = ["temporal_events", "memory_timeline", "event_metadata"]

            for table_name in tables_to_drop:
                try:
                    drop_query = f"DROP TABLE IF EXISTS {table_name} CASCADE"
                    await timescale_store.execute_query(drop_query)
                    logger.info(f"   Dropped table: {table_name}")
                except Exception as e:
                    logger.warning(f"   Table {table_name} drop failed: {e}")

            # Recreate schema
            await timescale_store.initialize_schema()

            self.flushed_stores.append("timescale_store")
            logger.info("   ✅ TimescaleDB flushed and schema recreated")

        except Exception as e:
            logger.error(f"   ❌ Failed to flush TimescaleDB: {e}")

    async def flush_audit_store(self):
        """Flush all data from audit store (RelationalDB)"""
        try:
            logger.info("🗑️  Flushing Audit Store (RelationalDB)...")
            audit_store = self.memory_engine.audit_store

            # Drop and recreate audit tables
            tables_to_drop = [
                "memory_audit_log",
                "access_logs",
                "system_events",
                "performance_metrics",
            ]

            for table_name in tables_to_drop:
                try:
                    drop_query = f"DROP TABLE IF EXISTS {table_name} CASCADE"
                    await audit_store.execute_query(drop_query)
                    logger.info(f"   Dropped table: {table_name}")
                except Exception as e:
                    logger.warning(f"   Table {table_name} drop failed: {e}")

            # Recreate schema
            await audit_store.initialize_schema()

            self.flushed_stores.append("audit_store")
            logger.info("   ✅ Audit store flushed and schema recreated")

        except Exception as e:
            logger.error(f"   ❌ Failed to flush audit store: {e}")

    async def verify_flush(self):
        """Verify that all stores are empty"""
        logger.info("🔍 Verifying flush results...")

        verification_results = {}

        try:
            # Check vector store
            vector_count = (
                await self.memory_engine.vector_store.get_total_vector_count()
            )
            verification_results["vector_store"] = vector_count == 0
            logger.info(f"   Vector store: {vector_count} vectors remaining")

            # Check document store
            doc_count = await self.memory_engine.doc_store.get_total_document_count()
            verification_results["document_store"] = doc_count == 0
            logger.info(f"   Document store: {doc_count} documents remaining")

            # Check graph store
            node_count = await self.memory_engine.graph_store.get_total_node_count()
            verification_results["graph_store"] = node_count == 0
            logger.info(f"   Graph store: {node_count} nodes remaining")

            # Check TimescaleDB
            event_count = (
                await self.memory_engine.timescale_store.get_total_event_count()
            )
            verification_results["timescale_store"] = event_count == 0
            logger.info(f"   TimescaleDB: {event_count} events remaining")

            # Check audit store
            audit_count = await self.memory_engine.audit_store.get_total_audit_count()
            verification_results["audit_store"] = audit_count == 0
            logger.info(f"   Audit store: {audit_count} audit logs remaining")

        except Exception as e:
            logger.warning(f"   Verification failed for some stores: {e}")

        return verification_results

    async def flush_all_databases(self, verify: bool = True):
        """Flush all databases and optionally verify"""
        logger.info("🚀 Starting database flush operation...")
        logger.info("=" * 60)

        # Initialize
        await self.initialize()

        # Flush all stores
        flush_operations = [
            self.flush_vector_store(),
            self.flush_document_store(),
            self.flush_graph_store(),
            self.flush_timescale_store(),
            self.flush_audit_store(),
        ]

        # Run all flush operations concurrently
        await asyncio.gather(*flush_operations, return_exceptions=True)

        # Verify if requested
        if verify:
            verification_results = await self.verify_flush()

            all_clean = all(verification_results.values())
            if all_clean:
                logger.info("   ✅ All databases successfully flushed and verified")
            else:
                logger.warning("   ⚠️  Some databases may still contain data")

        logger.info("=" * 60)
        logger.info(f"✅ Database flush completed!")
        logger.info(f"📊 Stores flushed: {len(self.flushed_stores)}")
        logger.info(f"🗂️  Flushed stores: {', '.join(self.flushed_stores)}")
        logger.info("\n💡 You can now start with a clean slate!")


async def main():
    """Main flush script"""
    print("🗑️  Database Flush Utility")
    print("This will remove ALL data from all memory stores!")
    print("=" * 60)

    # Auto-confirm for automated testing
    print("🚀 Auto-confirming database flush for testing...")
    confirmation = "yes"

    flusher = DatabaseFlusher()

    try:
        await flusher.flush_all_databases(verify=True)

        print("\n🎯 Next Steps:")
        print("   1. Run sample data script: python create_sample_data.py")
        print("   2. Test multi-hop reasoning: python test_multihop_reasoning.py")
        print("   3. Start fresh testing with: python test_qa_system.py")

    except Exception as e:
        logger.error(f"Flush operation failed: {e}")
        print(f"❌ Flush failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

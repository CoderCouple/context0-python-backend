"""
Script to create MongoDB indexes for performance optimization
"""
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.settings import settings
from app.db.indexes import create_indexes, get_index_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Create all indexes"""
    logger.info("Connecting to MongoDB...")

    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongodb_connection_string)
    db = client[settings.mongodb_database_name]

    try:
        # Test connection
        await db.command("ping")
        logger.info("Connected to MongoDB successfully")

        # Create indexes
        logger.info("Creating indexes...")
        await create_indexes(db)

        # Get index statistics
        logger.info("\nIndex Statistics:")
        stats = await get_index_stats(db)

        for collection, info in stats.items():
            logger.info(f"\n{collection}:")
            logger.info(f"  Document count: {info['document_count']:,}")
            logger.info(f"  Index count: {info['index_count']}")
            logger.info(f"  Total index size: {info['total_index_size']:,} bytes")
            logger.info(f"  Indexes: {', '.join(info['indexes'])}")

        logger.info("\n✅ All indexes created successfully!")

    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())

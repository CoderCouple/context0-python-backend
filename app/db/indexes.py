"""
MongoDB index creation and management
"""
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, TEXT

logger = logging.getLogger(__name__)


async def create_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create all necessary MongoDB indexes for optimal performance

    Note: MongoDB automatically skips index creation if the index already exists,
    so this is safe to run on every startup.
    """

    # Memory collection indexes
    memories_collection = db.memories

    # Composite index for user + timestamp queries (most common)
    await memories_collection.create_index(
        [("source_user_id", ASCENDING), ("created_at", DESCENDING)],
        name="user_time_idx",
    )
    logger.info("Created user_time_idx on memories collection")

    # Index for memory type filtering with user
    await memories_collection.create_index(
        [
            ("memory_type", ASCENDING),
            ("source_user_id", ASCENDING),
            ("created_at", DESCENDING),
        ],
        name="type_user_time_idx",
    )
    logger.info("Created type_user_time_idx on memories collection")

    # Text index for content search
    await memories_collection.create_index(
        [("input", TEXT), ("summary", TEXT), ("tags", TEXT)], name="content_text_idx"
    )
    logger.info("Created content_text_idx on memories collection")

    # Index for tag queries
    await memories_collection.create_index(
        [("tags", ASCENDING), ("source_user_id", ASCENDING)], name="tags_user_idx"
    )
    logger.info("Created tags_user_idx on memories collection")

    # Index for ID lookups (should be unique)
    await memories_collection.create_index("id", unique=True, name="memory_id_idx")
    logger.info("Created memory_id_idx on memories collection")

    # Chat sessions collection indexes
    sessions_collection = db.chat_sessions

    # Index for user's sessions sorted by update time
    await sessions_collection.create_index(
        [("user_id", ASCENDING), ("status", ASCENDING), ("updated_at", DESCENDING)],
        name="user_status_time_idx",
    )
    logger.info("Created user_status_time_idx on chat_sessions collection")

    # Index for session ID lookups
    await sessions_collection.create_index("id", unique=True, name="session_id_idx")
    logger.info("Created session_id_idx on chat_sessions collection")

    # Chat messages collection indexes
    messages_collection = db.chat_messages

    # Index for retrieving messages by session
    await messages_collection.create_index(
        [("session_id", ASCENDING), ("timestamp", ASCENDING)], name="session_time_idx"
    )
    logger.info("Created session_time_idx on chat_messages collection")

    # Index for message ID lookups
    await messages_collection.create_index("id", unique=True, name="message_id_idx")
    logger.info("Created message_id_idx on chat_messages collection")

    # LLM presets collection indexes
    presets_collection = db.llm_presets

    # Index for user's presets
    await presets_collection.create_index(
        [
            ("user_id", ASCENDING),
            ("is_default", DESCENDING),
            ("created_at", DESCENDING),
        ],
        name="user_default_time_idx",
    )
    logger.info("Created user_default_time_idx on llm_presets collection")

    # Index for preset ID lookups
    await presets_collection.create_index("id", unique=True, name="preset_id_idx")
    logger.info("Created preset_id_idx on llm_presets collection")

    # Tag extraction queue indexes
    tag_queue_collection = db.tag_extraction_queue

    # Index for pending items by user
    await tag_queue_collection.create_index(
        [("user_id", ASCENDING), ("status", ASCENDING), ("queued_at", ASCENDING)],
        name="user_status_queue_idx",
    )
    logger.info("Created user_status_queue_idx on tag_extraction_queue collection")

    # Memory extraction templates indexes
    templates_collection = db.memory_extraction_templates

    # Index for template lookups
    await templates_collection.create_index(
        [("memory_type", ASCENDING), ("version", DESCENDING)], name="type_version_idx"
    )
    logger.info("Created type_version_idx on memory_extraction_templates collection")

    # Chat extracted memories collection indexes
    extracted_memories_collection = db.chat_extracted_memories

    # Index for session queries
    await extracted_memories_collection.create_index(
        [("session_id", ASCENDING), ("extracted_at", DESCENDING)],
        name="session_time_idx",
    )
    logger.info("Created session_time_idx on chat_extracted_memories collection")

    # Index for message queries
    await extracted_memories_collection.create_index(
        "chat_message_id", name="message_id_idx"
    )
    logger.info("Created message_id_idx on chat_extracted_memories collection")

    # Index for user queries
    await extracted_memories_collection.create_index(
        [("user_id", ASCENDING), ("session_id", ASCENDING)], name="user_session_idx"
    )
    logger.info("Created user_session_idx on chat_extracted_memories collection")

    # Index for original memory reference
    await extracted_memories_collection.create_index(
        "original_memory_id", name="original_memory_idx"
    )
    logger.info("Created original_memory_idx on chat_extracted_memories collection")

    logger.info("All indexes created successfully")


async def drop_indexes(db: AsyncIOMotorDatabase) -> None:
    """Drop all custom indexes (useful for maintenance)"""

    collections = [
        "memories",
        "chat_sessions",
        "chat_messages",
        "llm_presets",
        "tag_extraction_queue",
        "memory_extraction_templates",
        "chat_extracted_memories",
    ]

    for collection_name in collections:
        collection = db[collection_name]
        # Drop all indexes except _id
        await collection.drop_indexes()
        logger.info(f"Dropped indexes for {collection_name} collection")


async def get_index_stats(db: AsyncIOMotorDatabase) -> dict:
    """Get statistics about index usage"""

    stats = {}

    collections = [
        "memories",
        "chat_sessions",
        "chat_messages",
        "llm_presets",
        "chat_extracted_memories",
    ]

    for collection_name in collections:
        collection = db[collection_name]

        # Get index information
        indexes = await collection.list_indexes().to_list(None)

        # Get collection stats
        coll_stats = await db.command("collStats", collection_name, indexDetails=True)

        stats[collection_name] = {
            "indexes": [idx["name"] for idx in indexes],
            "index_count": len(indexes),
            "total_index_size": coll_stats.get("totalIndexSize", 0),
            "document_count": coll_stats.get("count", 0),
        }

    return stats


# Index hints for specific queries
QUERY_HINTS = {
    "user_memories_recent": "user_time_idx",
    "user_memories_by_type": "type_user_time_idx",
    "memory_search": "content_text_idx",
    "memory_by_tags": "tags_user_idx",
    "session_list": "user_status_time_idx",
    "session_messages": "session_time_idx",
}

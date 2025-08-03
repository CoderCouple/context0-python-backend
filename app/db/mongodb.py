"""MongoDB database connection and dependencies"""
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.settings import settings

# Global MongoDB client instance
_mongodb_client: Optional[AsyncIOMotorClient] = None
_mongodb: Optional[AsyncIOMotorDatabase] = None


def get_mongodb_url() -> str:
    """Get MongoDB connection URL from settings"""
    return settings.mongodb_connection_string


def get_mongodb_name() -> str:
    """Get MongoDB database name from settings"""
    return settings.mongodb_database_name


async def connect_to_mongodb():
    """Create MongoDB connection with optimized pooling"""
    global _mongodb_client, _mongodb

    mongodb_url = get_mongodb_url()
    _mongodb_client = AsyncIOMotorClient(
        mongodb_url,
        maxPoolSize=100,
        minPoolSize=10,
        maxIdleTimeMS=30000,
        waitQueueTimeoutMS=5000,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        socketTimeoutMS=30000,
    )
    _mongodb = _mongodb_client[get_mongodb_name()]

    # Test connection
    await _mongodb.command("ping")
    print(f"Connected to MongoDB at {mongodb_url} with connection pooling")


async def close_mongodb_connection():
    """Close MongoDB connection"""
    global _mongodb_client

    if _mongodb_client:
        _mongodb_client.close()
        _mongodb_client = None
        print("Disconnected from MongoDB")


async def get_database() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance for dependency injection"""
    if _mongodb is None:
        await connect_to_mongodb()
    return _mongodb

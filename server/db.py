from typing import Optional

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from .config import settings


class MongoDB:
    client: Optional[AsyncIOMotorClient] = None


mongo = MongoDB()


async def connect_to_mongo(app: FastAPI) -> None:
    """Create Motor client and attach to app.state for reuse."""
    mongo.client = AsyncIOMotorClient(
        str(settings.mongo_url),
        maxPoolSize=settings.mongo_max_pool_size,
        serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
        tz_aware=True,
    )
    app.state.mongo_client = mongo.client


async def close_mongo_connection(app: FastAPI) -> None:
    if mongo.client:
        mongo.client.close()
        app.state.mongo_client = None


def get_collection() -> AsyncIOMotorCollection:
    if not mongo.client:
        raise RuntimeError("MongoDB client is not initialized")
    return mongo.client[settings.db_name][settings.collection_name]

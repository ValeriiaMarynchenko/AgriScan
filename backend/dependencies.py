from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorDatabase
from settings import settings
from fastapi import HTTPException, status

mongo_client: AsyncIOMotorClient | None = None
mongo_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo():
    """Створює асинхронне підключення до MongoDB при запуску FastAPI."""
    global mongo_client, mongo_db
    mongo_url = settings.MONGO_URI
    try:
        mongo_client = AsyncIOMotorClient(
            mongo_url,
            serverSelectionTimeoutMS=5000
        )

        await mongo_client.admin.command('ping')

        mongo_db = mongo_client[settings.MONGO_DB_NAME]
        print("INFO: Successfully connected to MongoDB.")

    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}")
        raise RuntimeError(f"MongoDB connection failed: {e}")

async def close_mongo_connection():
    """Закриває підключення до MongoDB при зупинці FastAPI."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("INFO: MongoDB connection closed.")


def get_db_client() -> AsyncIOMotorClient:
    """Залежність, що повертає клієнт MongoDB."""
    if mongo_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database client is unavailable."
        )
    return mongo_client

def get_database() -> AsyncIOMotorDatabase:
    """Залежність, що повертає базу даних MongoDB."""
    if mongo_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database client is unavailable."
        )
    return mongo_db


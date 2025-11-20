from venv import create

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.database import Database
from settings import settings

# Змінні для збереження клієнта та бази даних
mongo_client: AsyncIOMotorClient | None = None
mongo_db: Database | None = None


async def connect_to_mongo():
    """Створює асинхронне підключення до MongoDB при запуску FastAPI."""
    global mongo_client, mongo_db

    try:
        # Створюємо асинхронний клієнт Motor
        mongo_client = AsyncIOMotorClient(
            settings.MONGO_URL,
            serverSelectionTimeoutMS=5000  # Таймаут 5 секунд
        )

        # Перевіряємо підключення (це викине виняток, якщо не вдасться підключитися)
        await mongo_client.admin.command('ping')

        # Вибираємо базу даних
        mongo_db = mongo_client[settings.MONGO_DB_NAME]
        print("INFO: Successfully connected to MongoDB.")

    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}")
        # У виробничому середовищі тут варто обробити помилку більш суворо,
        # наприклад, зупинити додаток.


async def close_mongo_connection():
    """Закриває підключення до MongoDB при зупинці FastAPI."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("INFO: MongoDB connection closed.")


def get_db_client() -> AsyncIOMotorClient:
    """Залежність, що повертає клієнт MongoDB."""
    if mongo_client is None:
        raise Exception("MongoDB client is not initialized.")
    return mongo_client


def get_database() -> Database:
    """Залежність, що повертає базу даних MongoDB."""
    if mongo_db is None:
        raise Exception("MongoDB database is not initialized.")
    return mongo_db


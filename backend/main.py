from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dependencies import connect_to_mongo, close_mongo_connection, get_database
from settings import settings
from tasks import process_field_analysis
import uvicorn
# Створюємо екземпляр FastAPI
app = FastAPI(
    title="AgriScan FastAPI Backend",
    version="1.0.0",
    debug=settings.DEBUG,
    on_startup=[connect_to_mongo],
    on_shutdown=[close_mongo_connection]
)

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Дозволити всі джерела, як було у Django
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- HEALTH CHECK ROUTE ---
@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check(db: AsyncIOMotorClient = Depends(get_database)):
    """Перевірка стану додатку та підключення до бази даних."""
    try:
        # Спроба пінгу MongoDB
        await db.command('ping')

        # Перевірка Celery (можна додати більш складну логіку, наприклад, через redis-py)
        from tasks import celery_app
        celery_info = celery_app.control.inspect().stats()

        return {
            "status": "ok",
            "db": "MongoDB connected",
            "celery": "OK" if celery_info else "Celery workers unreachable",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: DB or Celery error: {e}"
        )


# --- EXAMPLE ROUTE (ROUTER для tasks) ---

@app.post("/api/analysis/start", tags=["Analysis"], status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(field_id: str, analysis_type: str = "NDVI"):
    """
    Запускає асинхронну задачу аналізу через Celery.
    Повертає ідентифікатор задачі.
    """
    # Запуск задачі Celery
    task = process_field_analysis.delay(field_id, analysis_type)

    return {
        "message": f"Analysis task for field {field_id} started.",
        "task_id": task.id,
        "status_url": f"/api/analysis/status/{task.id}"
    }


@app.get("/api/analysis/status/{task_id}", tags=["Analysis"])
async def get_analysis_status(task_id: str):
    """Перевіряє статус асинхронної задачі Celery."""
    from tasks import celery_app
    task_result = celery_app.AsyncResult(task_id)

    if task_result.ready():
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result,
        }

    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": None,
    }
#

# Примітка: Тут ви будете додавати роутери для 'apps.users', 'apps.fields' і т.д.
# app.include_router(user_router, prefix="/api/users", tags=["Users"])
# app.include_router(field_router, prefix="/api/fields", tags=["Fields"])


# --- ЗАПУСК ---
# Зазвичай запускається через Uvicorn, наприклад:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
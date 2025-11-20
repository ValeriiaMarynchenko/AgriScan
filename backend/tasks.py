from celery import Celery
from settings import settings
import time

# Ініціалізація Celery
celery_app = Celery(
    'agriscan_tasks',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Налаштування Celery (якщо потрібно)
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',  # Використовуємо налаштування з settings.py
    enable_utc=True,
)


@celery_app.task(name="process_field_analysis")
def process_field_analysis(field_id: str, analysis_type: str):
    """
    Асинхронна задача для виконання тривалого аналізу поля.

    Це імітує роботу, яку раніше виконувала б фонова задача Django.
    """
    print(f"INFO: Starting {analysis_type} analysis for field {field_id}...")

    # Імітація тривалого процесу (наприклад, обробка супутникових знімків)
    time.sleep(5)

    result = {"status": "completed", "field_id": field_id, "data": [12.5, 13.1]}

    print(f"INFO: Analysis for field {field_id} finished. Result: {result}")
    return result


# Приклад іншої задачі
@celery_app.task(name="send_welcome_email")
def send_welcome_email(user_email: str):
    """Задача для відправки електронного листа (наприклад, після реєстрації)."""
    print(f"INFO: Sending welcome email to {user_email}...")
    time.sleep(2)
    return {"status": "sent", "recipient": user_email}
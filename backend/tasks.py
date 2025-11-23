from celery import Celery
from settings import settings
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import smtplib

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
    timezone='UTC',
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


@celery_app.task(name="send_welcome_email")
def send_welcome_email(user_email: str):
    """
    Задача для відправки електронного листа через SMTP Gmail.
    """

    # 1. Створення об'єкта листа
    message = MIMEMultipart("alternative")
    message["Subject"] = "Ласкаво просимо до AgriScan! 🚜"
    message["From"] = settings.DEFAULT_FROM_EMAIL
    message["To"] = user_email

    # Створення тіла листа (можна додати HTML-версію)
    text = f"""
    Привіт!

    Дякуємо за реєстрацію на AgriScan. Тепер ви можете почати аналізувати свої поля.

    З повагою,
    Команда AgriScan
    """
    part1 = MIMEText(text, "plain")
    message.attach(part1)

    print(f"INFO: Attempting to send welcome email to {user_email} via Gmail SMTP...")

    # 2. Відправка через smtplib
    try:
        # Створюємо контекст SMTP
        with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
            # Запускаємо TLS-шифрування (обов'язково для порту 587)
            server.starttls()

            # Аутентифікація з використанням App Password
            server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)

            # Відправка
            server.sendmail(
                settings.DEFAULT_FROM_EMAIL,
                user_email,
                message.as_string()
            )

        print(f"INFO: Successfully sent welcome email to {user_email}")
        return {"status": "sent", "recipient": user_email}

    except Exception as e:
        # У випадку помилки з'єднання або аутентифікації
        error_message = f"ERROR: Failed to send email to {user_email}. Error: {e}"
        print(error_message)

        # Ви можете змусити Celery повторити спробу через певний час
        # raise self.retry(exc=e, countdown=60, max_retries=3)

        return {"status": "failed", "recipient": user_email, "error": str(e)}
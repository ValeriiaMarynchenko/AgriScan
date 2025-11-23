from pydantic_settings import BaseSettings, SettingsConfigDict
from datetime import timedelta
import os

# Визначаємо шлях до кореневої директорії проекту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    """
    Конфігурація проекту FastAPI.
    Всі значення завантажуються зі змінних оточення або використовують значення за замовчуванням.
    """

    # settings.py

    # --- Налаштування Пошти для Gmail ---
    EMAIL_HOST: str = 'smtp.gmail.com'  # SMTP-сервер Gmail
    EMAIL_PORT: int = 587  # Порт для TLS
    EMAIL_HOST_USER: str = 'agriscan.krnu@gmail.com'  # Повна адреса вашого облікового запису Gmail
    EMAIL_HOST_PASSWORD: str = 'vzzxauzogsswbrxp'  # Згенерований App Password
    EMAIL_USE_TLS: bool = True  # Використовувати шифрування TLS
    DEFAULT_FROM_EMAIL: str = 'agriscan.krnu@gmail.com'  # Адреса, від якої надсилаються листи

    # --- СЕКЦІЯ БЕЗПЕКИ ---
    SECRET_KEY: str = "fastapi-insecure-ri)hd7&x5_hj(ho1w9ai#1j-r!n&r1li)ju6!t)c#=&ovdcwd^"
    DEBUG: bool = True
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1", "*"]

    # --- БАЗА ДАНИХ (MongoDB) ---
    MONGO_DB_NAME: str = os.environ.get('MONGO_DB_NAME', 'agromonitoring_db')
    DB_HOST: str = os.environ.get('DB_HOST', 'localhost')
    DB_PORT: int = int(os.environ.get('DB_PORT', 27017))

    # НОВІ ПОЛЯ АВТЕНТИФІКАЦІЇ DB
    MONGO_USER: str = os.environ.get('MONGO_USER', 'mongo_user')
    MONGO_PASSWORD: str = os.environ.get('MONGO_PASSWORD', 'mongo_password')

    if MONGO_USER and MONGO_PASSWORD:
        MONGO_URI: str = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{DB_HOST}:{DB_PORT}/?authMechanism=DEFAULT"
    else:
        MONGO_URI: str = f"mongodb://{DB_HOST}:{DB_PORT}"

    # --- JWT / АВТЕНТИФІКАЦІЯ ---
    JWT_SECRET: str = SECRET_KEY
    ACCESS_TOKEN_LIFETIME: timedelta = timedelta(minutes=60)
    REFRESH_TOKEN_LIFETIME: timedelta = timedelta(days=1)

    # --- CELERY / REDIS ---
    CELERY_BROKER_URL: str = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
    CELERY_RESULT_BACKEND: str = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

    # --- ЗБЕРІГАННЯ ФАЙЛІВ (S3 / MinIO) ---
    USE_S3: bool = os.environ.get('USE_S3', 'False') == 'True'
    AWS_S3_CUSTOM_DOMAIN: str | None = os.environ.get('AWS_S3_CUSTOM_DOMAIN', None)
    AWS_STORAGE_BUCKET_NAME: str = os.environ.get('AWS_STORAGE_BUCKET_NAME', 'agriscan-media')

    # НОВІ ПОЛЯ ДЛЯ КОНФІГУРАЦІЇ S3/MinIO
    AWS_S3_ENDPOINT_URL: str = os.environ.get('AWS_S3_ENDPOINT_URL', 'minio:9000')  # Наприклад, для MinIO
    AWS_REGION: str = os.environ.get('AWS_REGION', 'us-east-1')
    AWS_S3_SECURE: bool = os.environ.get('AWS_S3_SECURE', 'False') == 'True'  # Використовувати HTTPS/SSL

    # Ключі, які вже були, але є обов'язковими для автентифікації
    AWS_ACCESS_KEY_ID: str = os.environ.get('MINIO_ACCESS_KEY', 'minio_access_key')
    AWS_SECRET_ACCESS_KEY: str = os.environ.get('MINIO_SECRET_KEY', 'minio_secret_key')

    model_config = SettingsConfigDict(
        env_file="../frontend/.env",
        extra="ignore"
    )


settings = Settings()
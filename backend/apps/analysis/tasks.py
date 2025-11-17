# apps/analysis/tasks.py
from celery import shared_task
from .models import AnalysisJob
from django.utils import timezone
import logging
logger = logging.getLogger(__name__)
# from ai_service_client import send_analysis_request # Клієнт для виклику ШІ-сервісу

@shared_task
def start_ai_analysis(job_id, image_url):
    """
    Асинхронне завдання для запуску аналізу зображення.
    """
    try:
        # 1. Оновлення статусу завдання в БД
        job = AnalysisJob.objects.get(id=job_id)
        job.status = 'PROCESSING'
        job.save()

        # 2. Виклик зовнішнього ШІ-сервісу
        result_data = {"something":"something, search this in backend/apps/analysis/tasks.py"}#send_analysis_request(image_url)

        # 3. Обробка результату та оновлення статусу
        job.result_url = result_data.get('output_url')
        job.status = 'COMPLETED'
        job.completed_at = timezone.now()
        job.save()

    except Exception as e:
        # Логіка обробки помилок
        logger.exception(f"Error during AI analysis for job {job_id}: {e}")
        if job:  # only update if job was successfully retrieved
            job.status = 'FAILED'
            job.save()

    # ... логування помилки
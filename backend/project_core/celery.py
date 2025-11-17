import os
from celery import Celery

# Встановлюємо змінну оточення Django для Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project_core.settings')

# Створюємо екземпляр Celery
app = Celery('project_core')

# Конфігурація Celery з налаштувань Django
# Всі налаштування Celery (CELERY_BROKER_URL, CELERY_RESULT_BACKEND)
# будуть братися з settings.py, якщо їх префікс CELERY_
app.config_from_object('django.conf:settings', namespace='CELERY')

# Автоматично виявляти та реєструвати завдання (tasks) з усіх
# додатків Django, які мають файл tasks.py
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
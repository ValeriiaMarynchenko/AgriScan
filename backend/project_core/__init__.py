# project_core/__init__.py
from .celery import app as celery_app

# Це забезпечує імпорт Celery при старті Django
__all__ = ('celery_app',)
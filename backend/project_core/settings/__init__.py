# project_core/settings/__init__.py
from .base import *

# Встановлюємо default settings
try:
    from .dev import *
except ImportError:
    # Якщо dev.py не знайдено, використовуємо лише base.py
    pass
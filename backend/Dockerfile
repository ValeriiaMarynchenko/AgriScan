#FROM ubuntu:latest
#LABEL authors="vlerk"
#
#ENTRYPOINT ["top", "-b"]

FROM python:3.12


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо requirements.txt та встановлюємо залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код
COPY . .

# Виставляємо порт, на якому працюватиме Django
EXPOSE 8000

# Команда запуску Gunicorn або Django runserver
# Для продакшну використовуйте Gunicorn:
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "project_core.wsgi:application"]
# Для розробки:
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000", "--settings=project_core.settings.dev"]
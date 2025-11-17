# apps/reports/urls.py
from django.urls import path
from views import generate_report_view, list_reports_view

urlpatterns = [
    # POST /api/v1/reports/generate/ - Запустити асинхронну генерацію звіту
    path('generate/', generate_report_view, name='generate-report'),

    # GET /api/v1/reports/list/?field_id=1 - Список доступних звітів
    path('list/', list_reports_view, name='list-reports'),
]
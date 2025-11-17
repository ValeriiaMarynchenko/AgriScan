from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import AnalysisJob
from .serializers import AnalysisJobSerializer
from tasks import start_ai_analysis


class AnalysisJobViewSet(viewsets.ModelViewSet):
    serializer_class = AnalysisJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Користувач бачить лише завдання, пов'язані з його полями
        return AnalysisJob.objects.filter(field__owner=self.request.user).order_by('-created_at')

    # Примусова дія для запуску аналізу (якщо це не робиться при POST)
    @action(detail=True, methods=['post'], url_path='run-analysis')
    def run_analysis(self, request, pk=None):
        job = self.get_object()

        if job.status == 'PENDING':
            # 1. Оновлюємо статус
            job.status = 'PROCESSING'
            job.save()

            # 2. Запускаємо асинхронне завдання Celery (потрібне налаштування Celery)
            start_ai_analysis.delay(job.id, job.image_url)

            return Response({'status': 'Аналіз запущено'}, status=status.HTTP_202_ACCEPTED)

        return Response({'status': f'Завдання вже в статусі {job.status}'}, status=status.HTTP_400_BAD_REQUEST)

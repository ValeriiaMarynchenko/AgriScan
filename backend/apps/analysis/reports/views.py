# apps/reports/views.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


# from .tasks import generate_report_task # Потрібне налаштування Celery

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_report_view(request):
    """Приймає запит на генерацію звіту та запускає асинхронне завдання."""
    field_id = request.data.get('field_id')
    report_type = request.data.get('report_type')

    if not field_id or not report_type:
        return Response({'error': 'Необхідні field_id та report_type.'}, status=status.HTTP_400_BAD_REQUEST)

    # 1. Перевірка, чи поле належить користувачу (рекомендовано)
    # try:
    #     Field.objects.get(id=field_id, owner=request.user)
    # except Field.DoesNotExist:
    #     return Response({'error': 'Поле не знайдено або доступ заборонено.'}, status=status.HTTP_403_FORBIDDEN)

    # 2. Запуск асинхронного завдання
    # generate_report_task.delay(field_id, report_type, request.user.id)

    return Response({'message': 'Генерацію звіту запущено, очікуйте сповіщення.'}, status=status.HTTP_202_ACCEPTED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_reports_view(request):
    """Повертає список доступних звітів для поля."""
    field_id = request.query_params.get('field_id')
    # ... логіка фільтрації та повернення звітів
    return Response({'reports': []})
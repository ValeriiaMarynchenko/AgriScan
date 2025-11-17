from rest_framework import viewsets, permissions
from .models import Field
from .serializers import FieldSerializer

class FieldViewSet(viewsets.ModelViewSet):
    """API для керування полями"""
    serializer_class = FieldSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Кожен користувач бачить лише свої поля
        return Field.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        # Автоматично призначаємо власника при створенні
        serializer.save(owner=self.request.user)
from django.shortcuts import render

# Цей файл може бути порожнім, якщо ви повністю покладаєтеся на Djoser
# для керування кінцевими точками (endpoints) користувачів.
# Djoser надає ViewSets, які автоматично підключаються через project_core/urls.py

from rest_framework import permissions, viewsets
from .models import CustomUser
from .serializers import CustomUserSerializer # Якщо ви хочете кастомний ViewSet

# Приклад, якщо б ви не використовували Djoser:
# class UserViewSet(viewsets.ModelViewSet):
#     queryset = CustomUser.objects.all()
#     serializer_class = CustomUserSerializer
#     permission_classes = [permissions.IsAuthenticated]

# Оскільки Djoser виконує цю роботу, залиште файл мінімальним.
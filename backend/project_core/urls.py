"""
URL configuration for project_core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from apps.fields.views import FieldViewSet
# ... імпорт інших ViewSet

router = DefaultRouter()
router.register(r'fields', FieldViewSet)
# router.register(r'users', UserViewSet)
# router.register(r'analysis', AnalysisViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    # Маршрути для JWT (для автентифікації)
    path('api/auth/', include('rest_framework_simplejwt.urls')),
    # path('api/auth/', include('djoser.urls')), # Альтернатива для повного Auth
]


# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]

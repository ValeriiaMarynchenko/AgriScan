"""
URL configuration for project_core project.
...
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from ..apps.fields.views import FieldViewSet # Assuming this path is correct

router = DefaultRouter()
router.register(r'fields', FieldViewSet)
# router.register(r'users', UserViewSet)
# router.register(r'analysis', AnalysisViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    # API endpoints for applications
    path('api/', include(router.urls)),

    # Djoser and JWT routes, all prefixed with 'api/auth/'
    path('api/auth/', include('djoser.urls')),
    path('auth/jwt/create/', include('djoser.urls.jwt')), # Routes for JWT (Token / Token Refresh)
    path('api/auth/', include('djoser.social.urls')), # Routes for OAuth

    path('api/reports/generate/', include('apps.reports.urls')),
]
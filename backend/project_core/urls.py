"""
URL configuration for project_core project.
...
"""
import djoser.views
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from ..apps.fields.views import FieldViewSet # Assuming this path is correct
from ..apps.analysis.views import AnalysisJobViewSet

router = DefaultRouter()
router.register(r'fields', FieldViewSet)
router.register(r'users', djoser.views.UserViewSet)
router.register(r'analysis', AnalysisJobViewSet)

urlpatterns = [

    path('admin/', admin.site.urls),
    path('', include(router.urls)),
    # API endpoints for applications
    path('api/', include(router.urls)),

    # Djoser and JWT routes, all prefixed with 'api/auth/'
    path('api/auth/', include('djoser.urls')),
    path('auth/jwt/create/', include('djoser.urls.jwt')), # Routes for JWT (Token / Token Refresh)
    path('api/auth/', include('djoser.social.urls')), # Routes for OAuth

    path('api/reports/generate/', include('apps.reports.urls')),
]
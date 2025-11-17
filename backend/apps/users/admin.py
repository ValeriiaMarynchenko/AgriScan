from django.contrib import admin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ("id", "email", "role", "organization_name", "is_staff", "is_active")
    search_fields = ("email", "organization_name")
    list_filter = ("role", "is_staff", "is_active")
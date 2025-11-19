from rest_framework import serializers
from .models import Field

class FieldSerializer(serializers.ModelSerializer):
    """Серіалізатор для полів (використовує стандартний ModelSerializer)"""
    class Meta:
        model = Field
        # Note: 'geometry' is replaced by 'geometry_data' (assuming you changed the model)
        fields = ('id', 'name', 'area_ha', 'owner', 'geometry_data')
        read_only_fields = ('owner', 'area_ha')
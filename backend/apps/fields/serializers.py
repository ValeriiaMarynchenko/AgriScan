from rest_framework_gis.serializers import GeoFeatureModelSerializer
from .models import Field

class FieldSerializer(GeoFeatureModelSerializer):
    """Серіалізатор для полів (використовує GeoJSON)"""
    class Meta:
        model = Field
        # Використовуйте 'geometry' як поле GeoJSON
        geo_field = 'geometry'
        fields = ('id', 'name', 'area_ha', 'owner')
        read_only_fields = ('owner', 'area_ha')
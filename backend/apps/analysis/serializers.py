from rest_framework import serializers
from .models import AnalysisJob

class AnalysisJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisJob
        fields = ('id', 'field', 'image_url', 'status', 'result_url', 'created_at', 'completed_at')
        read_only_fields = ('status', 'result_url', 'created_at', 'completed_at')
from rest_framework import serializers
from core.models import React
class ReactSerializer(serializers.ModelSerializer):
    class Meta:
        model = React
        fields = ['v1', 'v2', 'v3', 'v4']
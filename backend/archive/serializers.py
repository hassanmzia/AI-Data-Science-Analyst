from rest_framework import serializers
from .models import ArchivedProject, ArchivedAnalysis, ProjectTemplate


class ArchivedAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = ArchivedAnalysis
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class ArchivedProjectSerializer(serializers.ModelSerializer):
    archived_analyses = ArchivedAnalysisSerializer(many=True, read_only=True)

    class Meta:
        model = ArchivedProject
        fields = '__all__'
        read_only_fields = ['id', 'archived_at', 'updated_at', 'use_count']


class ArchivedProjectListSerializer(serializers.ModelSerializer):
    analysis_count = serializers.SerializerMethodField()

    class Meta:
        model = ArchivedProject
        fields = ['id', 'name', 'description', 'project_type', 'category',
                 'tags', 'tools_used', 'is_template', 'use_count',
                 'analysis_count', 'archived_at']

    def get_analysis_count(self, obj):
        return obj.archived_analyses.count()


class ProjectTemplateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectTemplate
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at', 'use_count']


class SearchArchiveSerializer(serializers.Serializer):
    query = serializers.CharField(help_text='Search query')
    category = serializers.CharField(required=False, default='')
    tags = serializers.ListField(child=serializers.CharField(), required=False, default=[])


class CreateFromArchiveSerializer(serializers.Serializer):
    archive_id = serializers.UUIDField()
    name = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, default='')

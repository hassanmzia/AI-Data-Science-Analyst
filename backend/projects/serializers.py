from rest_framework import serializers
from .models import Project, ProjectNote


class ProjectNoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectNote
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']


class ProjectSerializer(serializers.ModelSerializer):
    project_notes = ProjectNoteSerializer(many=True, read_only=True)
    analysis_count = serializers.SerializerMethodField()
    conversation_count = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_analysis_count(self, obj):
        return obj.analyses.count()

    def get_conversation_count(self, obj):
        return obj.conversations.count()


class ProjectListSerializer(serializers.ModelSerializer):
    dataset_count = serializers.SerializerMethodField()
    analysis_count = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = ['id', 'name', 'description', 'project_type', 'status',
                 'tags', 'dataset_count', 'analysis_count', 'created_at', 'updated_at']

    def get_dataset_count(self, obj):
        return obj.datasets.count()

    def get_analysis_count(self, obj):
        return obj.analyses.count()

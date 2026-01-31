from rest_framework import serializers
from .models import Dataset, DatabaseConnection, Document


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at', 'row_count',
                           'column_count', 'columns_info', 'file_size',
                           'preview_data', 'missing_values', 'dtypes', 'statistics']


class DatasetListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'description', 'source', 'file_format',
                 'row_count', 'column_count', 'file_size', 'tags',
                 'category', 'created_at', 'updated_at']


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    name = serializers.CharField(max_length=255, required=False)
    description = serializers.CharField(required=False, default='')
    tags = serializers.ListField(child=serializers.CharField(), required=False, default=[])


class KaggleImportSerializer(serializers.Serializer):
    dataset_ref = serializers.CharField(
        help_text='Kaggle dataset reference (e.g., "uciml/heart-disease")')
    name = serializers.CharField(max_length=255, required=False)
    description = serializers.CharField(required=False, default='')


class URLImportSerializer(serializers.Serializer):
    url = serializers.URLField()
    name = serializers.CharField(max_length=255, required=False)
    description = serializers.CharField(required=False, default='')


class DatabaseConnectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatabaseConnection
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']
        extra_kwargs = {'password': {'write_only': True}}


class DatabaseImportSerializer(serializers.Serializer):
    connection_id = serializers.UUIDField()
    query = serializers.CharField(help_text='SQL query or table name')
    name = serializers.CharField(max_length=255, required=False)


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at', 'is_indexed',
                           'chunk_count', 'collection_name', 'file_size']


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    name = serializers.CharField(max_length=255, required=False)
    description = serializers.CharField(required=False, default='')

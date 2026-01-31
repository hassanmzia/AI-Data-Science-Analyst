from django.contrib import admin
from .models import Dataset, DatabaseConnection, Document


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'source', 'file_format', 'row_count', 'column_count', 'created_at']
    list_filter = ['source', 'file_format', 'category']
    search_fields = ['name', 'description']


@admin.register(DatabaseConnection)
class DatabaseConnectionAdmin(admin.ModelAdmin):
    list_display = ['name', 'engine', 'host', 'database', 'is_active']
    list_filter = ['engine', 'is_active']


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['name', 'doc_type', 'is_indexed', 'chunk_count', 'created_at']
    list_filter = ['doc_type', 'is_indexed']

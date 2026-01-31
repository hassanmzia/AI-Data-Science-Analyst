import uuid
from django.db import models
from django.contrib.auth.models import User


class Dataset(models.Model):
    """Represents an uploaded or imported dataset."""

    SOURCE_CHOICES = [
        ('upload', 'File Upload'),
        ('kaggle', 'Kaggle Import'),
        ('database', 'Database Import'),
        ('url', 'URL Import'),
        ('sample', 'Sample Dataset'),
    ]

    FORMAT_CHOICES = [
        ('csv', 'CSV'),
        ('xlsx', 'Excel'),
        ('json', 'JSON'),
        ('parquet', 'Parquet'),
        ('sql', 'SQL Table'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='upload')
    file_format = models.CharField(max_length=20, choices=FORMAT_CHOICES, default='csv')
    file = models.FileField(upload_to='uploads/datasets/', null=True, blank=True)
    url = models.URLField(max_length=500, blank=True, default='')
    kaggle_ref = models.CharField(max_length=255, blank=True, default='',
                                  help_text='Kaggle dataset reference (e.g., user/dataset-name)')

    # Metadata
    row_count = models.IntegerField(null=True, blank=True)
    column_count = models.IntegerField(null=True, blank=True)
    columns_info = models.JSONField(default=dict, blank=True,
                                    help_text='Column names, types, and stats')
    file_size = models.BigIntegerField(null=True, blank=True, help_text='Size in bytes')
    preview_data = models.JSONField(default=dict, blank=True,
                                    help_text='First few rows for preview')
    missing_values = models.JSONField(default=dict, blank=True)
    dtypes = models.JSONField(default=dict, blank=True)
    statistics = models.JSONField(default=dict, blank=True,
                                  help_text='Descriptive statistics')

    # Tags and categorization
    tags = models.JSONField(default=list, blank=True)
    category = models.CharField(max_length=100, blank=True, default='')

    # Ownership
    owner = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                              related_name='datasets')
    is_public = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.source})"


class DatabaseConnection(models.Model):
    """External database connections for importing data."""

    ENGINE_CHOICES = [
        ('postgresql', 'PostgreSQL'),
        ('mysql', 'MySQL'),
        ('sqlite', 'SQLite'),
        ('mssql', 'MS SQL Server'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    engine = models.CharField(max_length=20, choices=ENGINE_CHOICES)
    host = models.CharField(max_length=255)
    port = models.IntegerField()
    database = models.CharField(max_length=255)
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)  # Encrypted in production
    is_active = models.BooleanField(default=True)
    owner = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.engine})"


class Document(models.Model):
    """Uploaded documents for RAG (PDFs, docs, etc)."""

    DOC_TYPE_CHOICES = [
        ('pdf', 'PDF'),
        ('docx', 'Word Document'),
        ('txt', 'Text File'),
        ('pptx', 'PowerPoint'),
        ('md', 'Markdown'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    doc_type = models.CharField(max_length=10, choices=DOC_TYPE_CHOICES)
    file = models.FileField(upload_to='uploads/documents/')
    file_size = models.BigIntegerField(null=True, blank=True)

    # RAG metadata
    is_indexed = models.BooleanField(default=False)
    chunk_count = models.IntegerField(null=True, blank=True)
    collection_name = models.CharField(max_length=255, blank=True, default='',
                                       help_text='ChromaDB collection name')
    embedding_model = models.CharField(max_length=100, default='text-embedding-3-large')

    owner = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.doc_type})"

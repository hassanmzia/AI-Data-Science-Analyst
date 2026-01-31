import io
import os
import logging
import pandas as pd
from uuid import uuid4

from django.conf import settings
from django.core.files.base import ContentFile
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from .models import Dataset, DatabaseConnection, Document
from .serializers import (
    DatasetSerializer, DatasetListSerializer, DatasetUploadSerializer,
    KaggleImportSerializer, URLImportSerializer,
    DatabaseConnectionSerializer, DatabaseImportSerializer,
    DocumentSerializer, DocumentUploadSerializer,
)
from .services import DatasetService, DocumentService

logger = logging.getLogger(__name__)


class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    filterset_fields = ['source', 'file_format', 'category', 'is_public']
    search_fields = ['name', 'description', 'tags']
    ordering_fields = ['created_at', 'name', 'row_count', 'file_size']

    def get_serializer_class(self):
        if self.action == 'list':
            return DatasetListSerializer
        return DatasetSerializer

    @action(detail=False, methods=['post'], serializer_class=DatasetUploadSerializer)
    def upload(self, request):
        """Upload a dataset file (CSV, Excel, JSON, Parquet)."""
        serializer = DatasetUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            dataset = DatasetService.create_from_upload(
                file=serializer.validated_data['file'],
                name=serializer.validated_data.get('name', ''),
                description=serializer.validated_data.get('description', ''),
                tags=serializer.validated_data.get('tags', []),
                owner=request.user if request.user.is_authenticated else None,
            )
            return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'], serializer_class=KaggleImportSerializer)
    def import_kaggle(self, request):
        """Import a dataset from Kaggle."""
        serializer = KaggleImportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            dataset = DatasetService.import_from_kaggle(
                dataset_ref=serializer.validated_data['dataset_ref'],
                name=serializer.validated_data.get('name', ''),
                description=serializer.validated_data.get('description', ''),
                owner=request.user if request.user.is_authenticated else None,
            )
            return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Kaggle import error: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'], serializer_class=URLImportSerializer)
    def import_url(self, request):
        """Import a dataset from a URL."""
        serializer = URLImportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            dataset = DatasetService.import_from_url(
                url=serializer.validated_data['url'],
                name=serializer.validated_data.get('name', ''),
                description=serializer.validated_data.get('description', ''),
                owner=request.user if request.user.is_authenticated else None,
            )
            return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"URL import error: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'], serializer_class=DatabaseImportSerializer)
    def import_database(self, request):
        """Import data from an external database."""
        serializer = DatabaseImportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            dataset = DatasetService.import_from_database(
                connection_id=serializer.validated_data['connection_id'],
                query=serializer.validated_data['query'],
                name=serializer.validated_data.get('name', ''),
                owner=request.user if request.user.is_authenticated else None,
            )
            return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Database import error: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def preview(self, request, pk=None):
        """Get a preview of the dataset (first 100 rows)."""
        dataset = self.get_object()
        try:
            preview = DatasetService.get_preview(dataset, limit=100)
            return Response(preview)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get descriptive statistics for the dataset."""
        dataset = self.get_object()
        try:
            stats = DatasetService.get_statistics(dataset)
            return Response(stats)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def column_info(self, request, pk=None):
        """Get detailed column information."""
        dataset = self.get_object()
        try:
            info = DatasetService.get_column_info(dataset)
            return Response(info)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def export(self, request, pk=None):
        """Export dataset to specified format."""
        dataset = self.get_object()
        fmt = request.data.get('format', 'csv')
        try:
            result = DatasetService.export_dataset(dataset, fmt)
            return Response(result)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DatabaseConnectionViewSet(viewsets.ModelViewSet):
    queryset = DatabaseConnection.objects.all()
    serializer_class = DatabaseConnectionSerializer
    filterset_fields = ['engine', 'is_active']

    @action(detail=True, methods=['post'])
    def test_connection(self, request, pk=None):
        """Test a database connection."""
        conn = self.get_object()
        try:
            success = DatasetService.test_db_connection(conn)
            return Response({'success': success, 'message': 'Connection successful'})
        except Exception as e:
            return Response({'success': False, 'message': str(e)},
                          status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def tables(self, request, pk=None):
        """List tables in the connected database."""
        conn = self.get_object()
        try:
            tables = DatasetService.list_tables(conn)
            return Response({'tables': tables})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    filterset_fields = ['doc_type', 'is_indexed']

    @action(detail=False, methods=['post'], serializer_class=DocumentUploadSerializer)
    def upload(self, request):
        """Upload and index a document for RAG."""
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            document = DocumentService.upload_and_index(
                file=serializer.validated_data['file'],
                name=serializer.validated_data.get('name', ''),
                description=serializer.validated_data.get('description', ''),
                owner=request.user if request.user.is_authenticated else None,
            )
            return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Document upload error: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def reindex(self, request, pk=None):
        """Re-index a document in the vector store."""
        document = self.get_object()
        try:
            DocumentService.index_document(document)
            return Response({'message': 'Document re-indexed successfully'})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def query(self, request, pk=None):
        """Query a specific document using RAG."""
        document = self.get_object()
        question = request.data.get('question', '')
        if not question:
            return Response({'error': 'Question is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            answer = DocumentService.query_document(document, question)
            return Response(answer)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

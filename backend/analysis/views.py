import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import AnalysisSession, Visualization, MLModel, HypothesisTest
from .serializers import (
    AnalysisSessionSerializer, AnalysisSessionListSerializer,
    RunEDASerializer, RunVisualizationSerializer, RunMLModelSerializer,
    RunHypothesisTestSerializer, RunSQLQuerySerializer,
    VisualizationSerializer, MLModelSerializer, MLModelListSerializer,
    HypothesisTestSerializer,
)
from .tasks import (
    run_eda_analysis, run_visualization_analysis,
    run_ml_model_training, run_hypothesis_test, run_sql_query,
)

logger = logging.getLogger(__name__)


class AnalysisSessionViewSet(viewsets.ModelViewSet):
    queryset = AnalysisSession.objects.all()
    serializer_class = AnalysisSessionSerializer
    filterset_fields = ['analysis_type', 'status', 'dataset']
    search_fields = ['name', 'description', 'query']

    def get_serializer_class(self):
        if self.action == 'list':
            return AnalysisSessionListSerializer
        return AnalysisSessionSerializer

    @action(detail=False, methods=['post'], serializer_class=RunEDASerializer)
    def run_eda(self, request):
        """Run EDA analysis using natural language."""
        serializer = RunEDASerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session = AnalysisSession.objects.create(
            name=serializer.validated_data.get('name', 'EDA Analysis'),
            analysis_type='eda',
            dataset_id=serializer.validated_data['dataset_id'],
            query=serializer.validated_data['query'],
            status='pending',
            owner=request.user if request.user.is_authenticated else None,
        )

        task = run_eda_analysis.delay(str(session.id))
        session.celery_task_id = task.id
        session.status = 'running'
        session.save()

        return Response(AnalysisSessionSerializer(session).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=False, methods=['post'], serializer_class=RunVisualizationSerializer)
    def run_visualization(self, request):
        """Create visualization using natural language."""
        serializer = RunVisualizationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session = AnalysisSession.objects.create(
            name=serializer.validated_data.get('name', 'Visualization'),
            analysis_type='visualization',
            dataset_id=serializer.validated_data['dataset_id'],
            query=serializer.validated_data['query'],
            parameters={'chart_type': serializer.validated_data.get('chart_type', 'auto')},
            status='pending',
            owner=request.user if request.user.is_authenticated else None,
        )

        task = run_visualization_analysis.delay(str(session.id))
        session.celery_task_id = task.id
        session.status = 'running'
        session.save()

        return Response(AnalysisSessionSerializer(session).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=False, methods=['post'], serializer_class=RunMLModelSerializer)
    def run_ml(self, request):
        """Train ML model using natural language."""
        serializer = RunMLModelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session = AnalysisSession.objects.create(
            name=serializer.validated_data.get('name', 'ML Model'),
            analysis_type='ml_model',
            dataset_id=serializer.validated_data['dataset_id'],
            query=serializer.validated_data['query'],
            parameters={
                'model_type': serializer.validated_data.get('model_type', 'auto'),
                'target_column': serializer.validated_data.get('target_column', ''),
            },
            status='pending',
            owner=request.user if request.user.is_authenticated else None,
        )

        task = run_ml_model_training.delay(str(session.id))
        session.celery_task_id = task.id
        session.status = 'running'
        session.save()

        return Response(AnalysisSessionSerializer(session).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=False, methods=['post'], serializer_class=RunHypothesisTestSerializer)
    def run_hypothesis_test(self, request):
        """Run hypothesis test using natural language."""
        serializer = RunHypothesisTestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session = AnalysisSession.objects.create(
            name=serializer.validated_data.get('name', 'Hypothesis Test'),
            analysis_type='hypothesis',
            dataset_id=serializer.validated_data['dataset_id'],
            query=serializer.validated_data['query'],
            parameters={'test_type': serializer.validated_data.get('test_type', 'auto')},
            status='pending',
            owner=request.user if request.user.is_authenticated else None,
        )

        task = run_hypothesis_test.delay(str(session.id))
        session.celery_task_id = task.id
        session.status = 'running'
        session.save()

        return Response(AnalysisSessionSerializer(session).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=False, methods=['post'], serializer_class=RunSQLQuerySerializer)
    def run_sql(self, request):
        """Run natural language SQL query."""
        serializer = RunSQLQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session = AnalysisSession.objects.create(
            name=serializer.validated_data.get('name', 'SQL Query'),
            analysis_type='sql_query',
            query=serializer.validated_data['query'],
            dataset_id=serializer.validated_data.get('dataset_id'),
            parameters={
                'connection_id': str(serializer.validated_data.get('connection_id', '')),
            },
            status='pending',
            owner=request.user if request.user.is_authenticated else None,
        )

        task = run_sql_query.delay(str(session.id))
        session.celery_task_id = task.id
        session.status = 'running'
        session.save()

        return Response(AnalysisSessionSerializer(session).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=True, methods=['get'])
    def status_check(self, request, pk=None):
        """Check the status of an analysis session."""
        session = self.get_object()
        return Response({
            'id': str(session.id),
            'status': session.status,
            'result': session.result if session.status == 'completed' else None,
            'error': session.error_message if session.status == 'failed' else None,
            'execution_time': session.execution_time,
        })


class VisualizationViewSet(viewsets.ModelViewSet):
    queryset = Visualization.objects.all()
    serializer_class = VisualizationSerializer
    filterset_fields = ['chart_type', 'dataset', 'analysis']


class MLModelViewSet(viewsets.ModelViewSet):
    queryset = MLModel.objects.all()
    filterset_fields = ['model_type', 'task_type', 'dataset']

    def get_serializer_class(self):
        if self.action == 'list':
            return MLModelListSerializer
        return MLModelSerializer

    @action(detail=True, methods=['post'])
    def predict(self, request, pk=None):
        """Run prediction using a trained model."""
        model = self.get_object()
        input_data = request.data.get('data', {})
        if not input_data:
            return Response({'error': 'Input data required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            from .services import MLService
            result = MLService.predict(model, input_data)
            return Response(result)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class HypothesisTestViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = HypothesisTest.objects.all()
    serializer_class = HypothesisTestSerializer
    filterset_fields = ['test_type', 'dataset']

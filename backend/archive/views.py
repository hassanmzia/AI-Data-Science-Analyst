import logging
from django.db.models import Q
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import ArchivedProject, ArchivedAnalysis, ProjectTemplate
from .serializers import (
    ArchivedProjectSerializer, ArchivedProjectListSerializer,
    ArchivedAnalysisSerializer, ProjectTemplateSerializer,
    SearchArchiveSerializer, CreateFromArchiveSerializer,
)

logger = logging.getLogger(__name__)


class ArchivedProjectViewSet(viewsets.ModelViewSet):
    queryset = ArchivedProject.objects.all()
    serializer_class = ArchivedProjectSerializer
    filterset_fields = ['project_type', 'category', 'is_template']
    search_fields = ['name', 'description', 'tags', 'search_keywords']

    def get_serializer_class(self):
        if self.action == 'list':
            return ArchivedProjectListSerializer
        return ArchivedProjectSerializer

    @action(detail=False, methods=['post'], serializer_class=SearchArchiveSerializer)
    def search(self, request):
        """Search archives using natural language."""
        serializer = SearchArchiveSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query = serializer.validated_data['query']
        category = serializer.validated_data.get('category', '')
        tags = serializer.validated_data.get('tags', [])

        qs = ArchivedProject.objects.all()

        # Text search
        if query:
            qs = qs.filter(
                Q(name__icontains=query) |
                Q(description__icontains=query) |
                Q(summary__icontains=query) |
                Q(methodology__icontains=query) |
                Q(lessons_learned__icontains=query)
            )

        if category:
            qs = qs.filter(category=category)

        if tags:
            for tag in tags:
                qs = qs.filter(tags__contains=[tag])

        return Response(ArchivedProjectListSerializer(qs[:50], many=True).data)

    @action(detail=True, methods=['post'])
    def create_project_from_archive(self, request, pk=None):
        """Create a new project based on an archived project."""
        archive = self.get_object()
        name = request.data.get('name', f"New from: {archive.name}")
        description = request.data.get('description', archive.description)

        from projects.models import Project
        project = Project.objects.create(
            name=name,
            description=description,
            project_type=archive.project_type,
            tags=archive.tags,
            notes=f"Created from archived project: {archive.name}\n\n"
                  f"Methodology: {archive.methodology}\n\n"
                  f"Lessons learned: {archive.lessons_learned}",
            config=archive.model_configs[0] if archive.model_configs else {},
        )

        # Increment use count
        archive.use_count += 1
        archive.save()

        from projects.serializers import ProjectSerializer
        return Response(ProjectSerializer(project).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def make_template(self, request, pk=None):
        """Convert an archived project into a reusable template."""
        archive = self.get_object()

        template = ProjectTemplate.objects.create(
            name=f"Template: {archive.name}",
            description=archive.description,
            category=archive.category,
            steps=[{
                'name': a.name,
                'type': a.analysis_type,
                'query': a.query,
                'description': a.description,
            } for a in archive.archived_analyses.all()],
            recommended_tools=archive.tools_used,
            sample_queries=[a.query for a in archive.archived_analyses.all() if a.query],
            source_archive=archive,
            tags=archive.tags,
        )

        archive.is_template = True
        archive.save()

        return Response(ProjectTemplateSerializer(template).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['get'])
    def similar(self, request, pk=None):
        """Find similar archived projects."""
        archive = self.get_object()

        # Simple tag-based similarity
        similar = ArchivedProject.objects.filter(
            category=archive.category
        ).exclude(id=archive.id)

        if archive.tags:
            tag_filter = Q()
            for tag in archive.tags:
                tag_filter |= Q(tags__contains=[tag])
            similar = similar.filter(tag_filter)

        return Response(ArchivedProjectListSerializer(similar[:10], many=True).data)


class ArchivedAnalysisViewSet(viewsets.ModelViewSet):
    queryset = ArchivedAnalysis.objects.all()
    serializer_class = ArchivedAnalysisSerializer
    filterset_fields = ['archived_project', 'analysis_type']
    search_fields = ['name', 'description', 'query']


class ProjectTemplateViewSet(viewsets.ModelViewSet):
    queryset = ProjectTemplate.objects.all()
    serializer_class = ProjectTemplateSerializer
    filterset_fields = ['category']
    search_fields = ['name', 'description', 'tags']

    @action(detail=True, methods=['post'])
    def use_template(self, request, pk=None):
        """Create a new project from a template."""
        template = self.get_object()
        name = request.data.get('name', f"Project from {template.name}")

        from projects.models import Project
        project = Project.objects.create(
            name=name,
            description=f"Created from template: {template.name}\n{template.description}",
            project_type=template.category if template.category != 'other' else 'general',
            tags=template.tags,
            config={
                'template_id': str(template.id),
                'steps': template.steps,
                'recommended_tools': template.recommended_tools,
            },
        )

        template.use_count += 1
        template.save()

        from projects.serializers import ProjectSerializer
        return Response(ProjectSerializer(project).data, status=status.HTTP_201_CREATED)

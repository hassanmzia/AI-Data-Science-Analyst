from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Project, ProjectNote
from .serializers import ProjectSerializer, ProjectListSerializer, ProjectNoteSerializer


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    filterset_fields = ['project_type', 'status']
    search_fields = ['name', 'description', 'tags']

    def get_serializer_class(self):
        if self.action == 'list':
            return ProjectListSerializer
        return ProjectSerializer

    @action(detail=True, methods=['post'])
    def generate_summary(self, request, pk=None):
        """Generate an AI summary of the project."""
        from agents.services import AgentOrchestrator
        project = self.get_object()

        analyses = project.analyses.filter(status='completed')
        analysis_summaries = []
        for a in analyses[:20]:
            analysis_summaries.append(f"- {a.name} ({a.analysis_type}): {str(a.result)[:200]}")

        prompt = f"""Summarize this data science project:
Project: {project.name}
Description: {project.description}
Type: {project.project_type}
Analyses completed:
{chr(10).join(analysis_summaries) if analysis_summaries else 'No analyses yet'}

Provide:
1. Executive summary (2-3 sentences)
2. Key findings (bullet points)
3. Recommendations"""

        try:
            orchestrator = AgentOrchestrator()
            from agents.models import Conversation
            dummy_conv = Conversation(assistant_type='general')
            result = orchestrator._handle_general(dummy_conv, prompt, [])

            project.summary = result['content']
            project.save()

            return Response({
                'summary': result['content'],
            })
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def archive(self, request, pk=None):
        """Archive the project."""
        project = self.get_object()
        project.status = 'archived'
        project.save()

        # Also create archive entry
        from archive.models import ArchivedProject
        ArchivedProject.objects.create(
            original_project=project,
            name=project.name,
            description=project.description,
            project_type=project.project_type,
            summary=project.summary,
            key_findings=project.key_findings,
            tags=project.tags,
            metadata={
                'analysis_count': project.analyses.count(),
                'dataset_count': project.datasets.count(),
                'archived_by': request.user.username if request.user.is_authenticated else 'system',
            },
        )

        return Response({'status': 'archived'})


class ProjectNoteViewSet(viewsets.ModelViewSet):
    queryset = ProjectNote.objects.all()
    serializer_class = ProjectNoteSerializer
    filterset_fields = ['project', 'note_type']

import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Conversation, Message, AgentConfig
from .serializers import (
    ConversationSerializer, ConversationListSerializer,
    MessageSerializer, ChatMessageSerializer, AgentConfigSerializer,
)
from .services import AgentOrchestrator

logger = logging.getLogger(__name__)


class ConversationViewSet(viewsets.ModelViewSet):
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer
    filterset_fields = ['assistant_type', 'is_archived', 'dataset', 'project']
    search_fields = ['title']

    def get_serializer_class(self):
        if self.action == 'list':
            return ConversationListSerializer
        return ConversationSerializer

    @action(detail=True, methods=['post'], serializer_class=ChatMessageSerializer)
    def chat(self, request, pk=None):
        """Send a message and get AI response."""
        conversation = self.get_object()
        serializer = ChatMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user_message = serializer.validated_data['message']
        dataset_id = serializer.validated_data.get('dataset_id')
        document_id = serializer.validated_data.get('document_id')

        # Save user message
        user_msg = Message.objects.create(
            conversation=conversation,
            role='user',
            content=user_message,
        )

        # Update dataset context if provided
        if dataset_id:
            from datasets.models import Dataset
            try:
                conversation.dataset_id = dataset_id
                conversation.save()
            except Dataset.DoesNotExist:
                pass

        try:
            orchestrator = AgentOrchestrator()
            response_data = orchestrator.process_message(
                conversation=conversation,
                message=user_message,
                dataset_id=dataset_id,
                document_id=document_id,
            )

            # Save assistant message
            assistant_msg = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=response_data.get('content', ''),
                agent_name=response_data.get('agent_name', ''),
                tool_calls=response_data.get('tool_calls', []),
                code_blocks=response_data.get('code_blocks', []),
                visualizations=response_data.get('visualizations', []),
                metadata=response_data.get('metadata', {}),
            )

            return Response({
                'user_message': MessageSerializer(user_msg).data,
                'assistant_message': MessageSerializer(assistant_msg).data,
            })

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=f"I encountered an error: {str(e)}. Please try again.",
                metadata={'error': True},
            )
            return Response({
                'user_message': MessageSerializer(user_msg).data,
                'assistant_message': MessageSerializer(error_msg).data,
            }, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def archive(self, request, pk=None):
        """Archive a conversation."""
        conversation = self.get_object()
        conversation.is_archived = True
        conversation.save()
        return Response({'status': 'archived'})

    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get all messages in a conversation."""
        conversation = self.get_object()
        messages = conversation.messages.all()
        return Response(MessageSerializer(messages, many=True).data)


class AgentConfigViewSet(viewsets.ModelViewSet):
    queryset = AgentConfig.objects.filter(is_active=True)
    serializer_class = AgentConfigSerializer
    filterset_fields = ['agent_type', 'is_active']

import json
import logging
from channels.generic.websocket import AsyncJsonWebsocketConsumer

logger = logging.getLogger(__name__)


class ChatConsumer(AsyncJsonWebsocketConsumer):
    """WebSocket consumer for real-time chat with AI agents."""

    async def connect(self):
        self.conversation_id = self.scope['url_route']['kwargs']['conversation_id']
        self.room_group = f'chat_{self.conversation_id}'

        await self.channel_layer.group_add(self.room_group, self.channel_name)
        await self.accept()

        await self.send_json({
            'type': 'connection_established',
            'conversation_id': self.conversation_id,
        })

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group, self.channel_name)

    async def receive_json(self, content):
        message = content.get('message', '')
        dataset_id = content.get('dataset_id')
        document_id = content.get('document_id')

        # Send typing indicator
        await self.send_json({
            'type': 'typing',
            'is_typing': True,
        })

        try:
            from .services import AgentOrchestrator
            from .models import Conversation, Message
            from channels.db import database_sync_to_async

            @database_sync_to_async
            def process():
                conversation = Conversation.objects.get(id=self.conversation_id)

                # Save user message
                Message.objects.create(
                    conversation=conversation,
                    role='user',
                    content=message,
                )

                orchestrator = AgentOrchestrator()
                response_data = orchestrator.process_message(
                    conversation=conversation,
                    message=message,
                    dataset_id=dataset_id,
                    document_id=document_id,
                )

                # Save assistant message
                msg = Message.objects.create(
                    conversation=conversation,
                    role='assistant',
                    content=response_data.get('content', ''),
                    agent_name=response_data.get('agent_name', ''),
                    tool_calls=response_data.get('tool_calls', []),
                    code_blocks=response_data.get('code_blocks', []),
                    visualizations=response_data.get('visualizations', []),
                    metadata=response_data.get('metadata', {}),
                )
                return response_data

            result = await process()

            await self.send_json({
                'type': 'message',
                'role': 'assistant',
                'content': result.get('content', ''),
                'agent_name': result.get('agent_name', ''),
                'code_blocks': result.get('code_blocks', []),
                'visualizations': result.get('visualizations', []),
                'metadata': result.get('metadata', {}),
            })

        except Exception as e:
            logger.error(f"WebSocket chat error: {e}")
            await self.send_json({
                'type': 'error',
                'message': str(e),
            })
        finally:
            await self.send_json({
                'type': 'typing',
                'is_typing': False,
            })


class AnalysisConsumer(AsyncJsonWebsocketConsumer):
    """WebSocket consumer for real-time analysis status updates."""

    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group = f'analysis_{self.session_id}'

        await self.channel_layer.group_add(self.room_group, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group, self.channel_name)

    async def analysis_update(self, event):
        await self.send_json(event['data'])

from rest_framework import serializers
from .models import Conversation, Message, AgentConfig


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()


class ConversationListSerializer(serializers.ModelSerializer):
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'assistant_type', 'dataset', 'project',
                 'message_count', 'last_message', 'is_archived',
                 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last = obj.messages.order_by('-created_at').first()
        if last:
            return {'role': last.role, 'content': last.content[:200], 'created_at': last.created_at}
        return None


class ChatMessageSerializer(serializers.Serializer):
    message = serializers.CharField()
    dataset_id = serializers.UUIDField(required=False)
    document_id = serializers.UUIDField(required=False)
    tools = serializers.ListField(child=serializers.CharField(), required=False, default=[])


class AgentConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentConfig
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']

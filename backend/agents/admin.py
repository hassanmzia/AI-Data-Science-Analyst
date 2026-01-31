from django.contrib import admin
from .models import Conversation, Message, AgentConfig


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['title', 'assistant_type', 'dataset', 'is_archived', 'created_at']
    list_filter = ['assistant_type', 'is_archived']


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'role', 'agent_name', 'created_at']
    list_filter = ['role', 'agent_name']


@admin.register(AgentConfig)
class AgentConfigAdmin(admin.ModelAdmin):
    list_display = ['display_name', 'agent_type', 'model', 'is_active']
    list_filter = ['agent_type', 'is_active']

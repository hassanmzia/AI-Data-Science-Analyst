import uuid
from django.db import models
from django.contrib.auth.models import User


class Conversation(models.Model):
    """A conversation thread with the AI assistant."""

    ASSISTANT_TYPES = [
        ('data_analyst', 'Data Analyst'),
        ('data_scientist', 'Data Scientist'),
        ('sql_expert', 'SQL Expert'),
        ('ml_engineer', 'ML Engineer'),
        ('rag_assistant', 'RAG Assistant'),
        ('general', 'General Assistant'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, default='New Conversation')
    assistant_type = models.CharField(max_length=20, choices=ASSISTANT_TYPES, default='general')

    dataset = models.ForeignKey('datasets.Dataset', on_delete=models.SET_NULL,
                                null=True, blank=True, related_name='conversations')
    project = models.ForeignKey('projects.Project', on_delete=models.SET_NULL,
                                null=True, blank=True, related_name='conversations')

    # Context
    system_prompt = models.TextField(blank=True, default='')
    context = models.JSONField(default=dict, blank=True,
                               help_text='Additional context for the conversation')

    owner = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    is_archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.title} ({self.assistant_type})"


class Message(models.Model):
    """A message in a conversation."""

    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
        ('tool', 'Tool Output'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,
                                     related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()

    # Agent metadata
    agent_name = models.CharField(max_length=100, blank=True, default='')
    tool_calls = models.JSONField(default=list, blank=True)
    tool_results = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True,
                                help_text='Token usage, latency, etc.')

    # Attachments
    code_blocks = models.JSONField(default=list, blank=True)
    visualizations = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"


class AgentConfig(models.Model):
    """Configuration for different AI agents."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    display_name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    agent_type = models.CharField(max_length=50)

    # LLM config
    model = models.CharField(max_length=100, default='gpt-4o-mini')
    temperature = models.FloatField(default=0.0)
    max_tokens = models.IntegerField(default=4096)
    system_prompt = models.TextField(blank=True, default='')

    # Tools
    tools = models.JSONField(default=list, blank=True,
                             help_text='List of tools available to this agent')
    mcp_tools = models.JSONField(default=list, blank=True,
                                 help_text='MCP tool names')

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.display_name

import uuid
from django.db import models
from django.contrib.auth.models import User


class Project(models.Model):
    """A data science / analytics project that groups analyses together."""

    STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('on_hold', 'On Hold'),
        ('archived', 'Archived'),
    ]

    TYPE_CHOICES = [
        ('eda', 'Exploratory Data Analysis'),
        ('ml', 'Machine Learning'),
        ('analytics', 'Data Analytics'),
        ('reporting', 'Reporting'),
        ('research', 'Research'),
        ('general', 'General'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    project_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='general')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')

    # Related datasets
    datasets = models.ManyToManyField('datasets.Dataset', blank=True, related_name='projects')
    documents = models.ManyToManyField('datasets.Document', blank=True, related_name='projects')

    # Configuration
    config = models.JSONField(default=dict, blank=True)
    tags = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True, default='')

    # Results & outputs
    summary = models.TextField(blank=True, default='',
                               help_text='AI-generated project summary')
    key_findings = models.JSONField(default=list, blank=True)

    owner = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                              related_name='projects')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.name} ({self.status})"


class ProjectNote(models.Model):
    """Notes and annotations for a project."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='project_notes')
    title = models.CharField(max_length=255)
    content = models.TextField()
    note_type = models.CharField(max_length=50, default='general',
                                 choices=[('general', 'General'), ('finding', 'Finding'),
                                          ('todo', 'To Do'), ('decision', 'Decision')])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} ({self.project.name})"

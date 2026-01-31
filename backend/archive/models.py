import uuid
from django.db import models
from django.contrib.auth.models import User


class ArchivedProject(models.Model):
    """Archived data science / analytics projects for future reuse."""

    CATEGORY_CHOICES = [
        ('eda', 'Exploratory Data Analysis'),
        ('ml_classification', 'ML Classification'),
        ('ml_regression', 'ML Regression'),
        ('ml_clustering', 'ML Clustering'),
        ('nlp', 'Natural Language Processing'),
        ('time_series', 'Time Series Analysis'),
        ('statistical_analysis', 'Statistical Analysis'),
        ('data_cleaning', 'Data Cleaning'),
        ('feature_engineering', 'Feature Engineering'),
        ('reporting', 'Reporting & Dashboards'),
        ('other', 'Other'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_project = models.ForeignKey('projects.Project', on_delete=models.SET_NULL,
                                          null=True, blank=True, related_name='archives')

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    project_type = models.CharField(max_length=20, default='general')
    category = models.CharField(max_length=30, choices=CATEGORY_CHOICES, default='other')

    # Preserved content
    summary = models.TextField(blank=True, default='')
    key_findings = models.JSONField(default=list, blank=True)
    methodology = models.TextField(blank=True, default='',
                                   help_text='Description of methodology used')
    code_snippets = models.JSONField(default=list, blank=True,
                                     help_text='Reusable code snippets from the project')
    analysis_results = models.JSONField(default=dict, blank=True,
                                        help_text='Preserved analysis results')
    model_configs = models.JSONField(default=list, blank=True,
                                     help_text='ML model configurations that worked')

    # Reusability metadata
    tags = models.JSONField(default=list, blank=True)
    tools_used = models.JSONField(default=list, blank=True,
                                   help_text='Tools and libraries used')
    dataset_description = models.TextField(blank=True, default='',
                                           help_text='Description of original dataset')
    lessons_learned = models.TextField(blank=True, default='')

    # Search and discovery
    search_keywords = models.JSONField(default=list, blank=True)
    similarity_embedding = models.JSONField(default=list, blank=True,
                                            help_text='Embedding for semantic search')

    metadata = models.JSONField(default=dict, blank=True)

    owner = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    is_template = models.BooleanField(default=False,
                                       help_text='Can this be used as a template for new projects?')
    use_count = models.IntegerField(default=0, help_text='Number of times reused')

    archived_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-archived_at']
        verbose_name_plural = 'Archived projects'

    def __str__(self):
        return f"[Archive] {self.name}"


class ArchivedAnalysis(models.Model):
    """Individual archived analysis results for granular reuse."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    archived_project = models.ForeignKey(ArchivedProject, on_delete=models.CASCADE,
                                          related_name='archived_analyses')

    name = models.CharField(max_length=255)
    analysis_type = models.CharField(max_length=20)
    description = models.TextField(blank=True, default='')

    # Preserved content
    query = models.TextField(blank=True, default='')
    code = models.TextField(blank=True, default='')
    result = models.JSONField(default=dict, blank=True)
    visualizations = models.JSONField(default=list, blank=True)

    tags = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Archived analyses'

    def __str__(self):
        return f"[Archive] {self.name} ({self.analysis_type})"


class ProjectTemplate(models.Model):
    """Reusable project templates derived from archived projects."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField()
    category = models.CharField(max_length=30, default='other')

    # Template definition
    steps = models.JSONField(default=list,
                             help_text='Ordered list of analysis steps')
    recommended_tools = models.JSONField(default=list)
    sample_queries = models.JSONField(default=list,
                                      help_text='Example natural language queries')
    config = models.JSONField(default=dict, blank=True)

    source_archive = models.ForeignKey(ArchivedProject, on_delete=models.SET_NULL,
                                        null=True, blank=True)

    use_count = models.IntegerField(default=0)
    rating = models.FloatField(default=0.0)
    tags = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-use_count', '-rating']

    def __str__(self):
        return f"[Template] {self.name}"

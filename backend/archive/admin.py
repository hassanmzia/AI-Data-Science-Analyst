from django.contrib import admin
from .models import ArchivedProject, ArchivedAnalysis, ProjectTemplate


@admin.register(ArchivedProject)
class ArchivedProjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'project_type', 'category', 'is_template', 'use_count', 'archived_at']
    list_filter = ['project_type', 'category', 'is_template']
    search_fields = ['name', 'description', 'tags']


@admin.register(ArchivedAnalysis)
class ArchivedAnalysisAdmin(admin.ModelAdmin):
    list_display = ['name', 'analysis_type', 'archived_project', 'created_at']
    list_filter = ['analysis_type']


@admin.register(ProjectTemplate)
class ProjectTemplateAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'use_count', 'rating', 'created_at']
    list_filter = ['category']

from django.contrib import admin
from .models import Project, ProjectNote


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'project_type', 'status', 'created_at', 'updated_at']
    list_filter = ['project_type', 'status']
    search_fields = ['name', 'description']


@admin.register(ProjectNote)
class ProjectNoteAdmin(admin.ModelAdmin):
    list_display = ['title', 'project', 'note_type', 'created_at']
    list_filter = ['note_type']

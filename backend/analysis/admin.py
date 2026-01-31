from django.contrib import admin
from .models import AnalysisSession, Visualization, MLModel, HypothesisTest


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'analysis_type', 'status', 'dataset', 'execution_time', 'created_at']
    list_filter = ['analysis_type', 'status']
    search_fields = ['name', 'query']


@admin.register(Visualization)
class VisualizationAdmin(admin.ModelAdmin):
    list_display = ['name', 'chart_type', 'dataset', 'created_at']
    list_filter = ['chart_type']


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'task_type', 'dataset', 'created_at']
    list_filter = ['model_type', 'task_type']


@admin.register(HypothesisTest)
class HypothesisTestAdmin(admin.ModelAdmin):
    list_display = ['name', 'test_type', 'p_value', 'reject_null', 'created_at']
    list_filter = ['test_type', 'reject_null']

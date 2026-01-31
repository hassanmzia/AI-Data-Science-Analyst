import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

app = Celery('core')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

app.conf.task_routes = {
    'agents.tasks.*': {'queue': 'default'},
    'analysis.tasks.run_ml_*': {'queue': 'ml_tasks'},
    'analysis.tasks.run_eda_*': {'queue': 'analysis_tasks'},
    'analysis.tasks.run_visualization_*': {'queue': 'analysis_tasks'},
}

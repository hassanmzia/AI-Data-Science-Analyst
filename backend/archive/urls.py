from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ArchivedProjectViewSet, ArchivedAnalysisViewSet, ProjectTemplateViewSet

router = DefaultRouter()
router.register(r'projects', ArchivedProjectViewSet)
router.register(r'analyses', ArchivedAnalysisViewSet)
router.register(r'templates', ProjectTemplateViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    AnalysisSessionViewSet, VisualizationViewSet,
    MLModelViewSet, HypothesisTestViewSet,
)

router = DefaultRouter()
router.register(r'sessions', AnalysisSessionViewSet)
router.register(r'visualizations', VisualizationViewSet)
router.register(r'models', MLModelViewSet)
router.register(r'hypothesis-tests', HypothesisTestViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

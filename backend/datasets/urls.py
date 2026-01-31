from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DatasetViewSet, DatabaseConnectionViewSet, DocumentViewSet

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet)
router.register(r'connections', DatabaseConnectionViewSet)
router.register(r'documents', DocumentViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

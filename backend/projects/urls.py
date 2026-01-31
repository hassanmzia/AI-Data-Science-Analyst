from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ProjectViewSet, ProjectNoteViewSet

router = DefaultRouter()
router.register(r'projects', ProjectViewSet)
router.register(r'notes', ProjectNoteViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

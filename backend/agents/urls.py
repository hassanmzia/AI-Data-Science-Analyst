from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ConversationViewSet, AgentConfigViewSet

router = DefaultRouter()
router.register(r'conversations', ConversationViewSet)
router.register(r'configs', AgentConfigViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

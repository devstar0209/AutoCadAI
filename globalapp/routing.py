from django.urls import path
from .consumer import PDFProcessingConsumer

websocket_urlpatterns = [
    path("ws/pdf-status/", PDFProcessingConsumer.as_asgi()),
]

from django.urls import path
from .views import upload_pdf, create_session, stripe_webhook, classify_csi_code

urlpatterns = [
    path('upload/', upload_pdf, name='upload_pdf'),
    path('create_session/', create_session, name='create_session'),
    path('stripe-webhook/', stripe_webhook, name='stripe_webhook'),
    path('classify_csi_code/', classify_csi_code, name='classify_csi_code')
]
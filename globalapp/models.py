import os
from django.db import models

def pdf_upload_path(instance, filename):
    file_name_without_ext = os.path.splitext(filename)[0]
    cleaned_name = file_name_without_ext.replace("%", "")
    return f'pdfs/{cleaned_name}/{filename}'

# Create your models here.
class UploadedPDF(models.Model):
    title = models.CharField(max_length=255)
    pdf_file = models.FileField(upload_to=pdf_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
class UploadSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    status = models.CharField(max_length=50, default='awaiting_upload')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional: track user, file, price, etc.
    pdf_uploaded = models.BooleanField(default=False)
    excel_path = models.CharField(max_length=500, null=True, blank=True)
    output_pdf_path = models.CharField(max_length=500, null=True, blank=True)
    pdf_path = models.CharField(max_length=500, null=True, blank=True)
    calculated_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    payment_completed = models.BooleanField(default=False)
    country = models.CharField(max_length=100, null=True, blank=True)
    currency = models.CharField(max_length=10, null=True, blank=True)
    unit = models.CharField(max_length=50, null=True, blank=True)
    project_title = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"Session {self.session_id} - {self.status}"
class Cost(models.Model):
    cost_per_page = models.DecimalField(max_digits=10, decimal_places=0)

    def __str__(self):
        return f"{self.cost_per_page}"
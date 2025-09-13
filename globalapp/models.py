import os
from django.db import models

def pdf_upload_path(instance, filename):
    file_name_without_ext = os.path.splitext(filename)[0]
    return f'pdfs/{file_name_without_ext}/{filename}'

# Create your models here.
class UploadedPDF(models.Model):
    title = models.CharField(max_length=255)
    pdf_file = models.FileField(upload_to=pdf_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
    
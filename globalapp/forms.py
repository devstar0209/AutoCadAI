from django import forms
from .models import UploadedPDF

class PDFUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedPDF
        fields = ['title', 'pdf_file']
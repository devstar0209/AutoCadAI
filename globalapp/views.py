import os
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .forms import PDFUploadForm
from django.http import JsonResponse
from django.conf import settings
from .task import start_pdf_processing
# Create your views here.

@csrf_exempt
def upload_pdf(request):
    try:
        if request.method == 'POST':
            print("request arrived===>")
            form = PDFUploadForm(request.POST, request.FILES)
            if form.is_valid():
                pdf_instance = form.save()
                file_path = pdf_instance.pdf_file.path
                directory_path = os.path.dirname(file_path)
                # if os.path.isabs(directory_path):
                #     directory_path = os.path.relpath(directory_path, os.path.abspath(os.sep))
                # directory_path = os.path.join(settings.BASE_DIR, directory_path)
                print("base directory", file_path, directory_path)
                excel_path = file_path.replace(".pdf", ".xlsx")
                # excel_path = os.path.join(os.path.normpath(settings.BASE_DIR), os.path.normpath(excel_path))
                output_pdf_path = file_path.replace(".pdf", "_cost.pdf")
                # output_pdf_path = os.path.join(os.path.normpath(settings.BASE_DIR), os.path.normpath(directory_path))
                # image_path = os.path.join(os.path.normpath(settings.BASE_DIR), os.path.normpath(directory_path))
                if not os.path.exists(directory_path):
                    return JsonResponse({"error": "PDF directory not found."}, status=400)
                start_pdf_processing(file_path, excel_path, output_pdf_path)
                # print("image will be saved here!!", image_path) 
                return JsonResponse({
                "success": True,
                "message": "PDF uploaded successfully, You will be notified when cost estimation will be done.",
                "file_url": pdf_instance.pdf_file.url,  # Public URL
                "file_path": pdf_instance.pdf_file.path  # Full file path on the server
                }, status=200)
    except Exception as e:
        return JsonResponse({"success" : False, "error": str(e)})    
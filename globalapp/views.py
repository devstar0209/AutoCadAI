import os
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .forms import PDFUploadForm
from django.http import JsonResponse
from django.conf import settings
from .task import start_pdf_processing, extract_json_from_response
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from django.db import connection
from uuid import uuid4
from .models import UploadSession, Cost
import fitz
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import stripe
import openai
# Create your views here.
# Set a limit for parallel processing
MAX_CONCURRENT_PDFS = 1  # Control how many PDFs are processed at once
MAX_THREADS_PER_PDF = 2  # Limit threads per PDF to avoid CPU overload
API_KEY = ""
client = openai.OpenAI(api_key=API_KEY)
# Global thread pool for processing PDFs
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PDFS)
stripe.api_key = ''

@csrf_exempt
def upload_pdf(request):
    try:
        if request.method == 'POST':
            print("request arrived===>")
            form = PDFUploadForm(request.POST, request.FILES)
            country= request.POST.get('country')
            currency = request.POST.get('currency')
            # country = "Jamaica"
            # currency = "JMD"
            session_id = request.POST.get('session_id')
            project_title = request.POST.get('project_title')
            print("country===>", country, currency, session_id, project_title)
            if form.is_valid():
                pdf_instance = form.save()
                file_path = pdf_instance.pdf_file.path
                directory_path = os.path.dirname(file_path)
                
                print("base directory", file_path, directory_path)
                doc = fitz.open(file_path)
                total_pages = len(doc)
                print("page_count===>",total_pages)
                cost_data = Cost.objects.first()
                cost = cost_data.cost_per_page*100 if cost_data else 1000
                print("cost_per_page===>", cost)
                excel_path = file_path.replace(".pdf", ".xlsx")
                output_pdf_path = file_path.replace(".pdf", "_cost.pdf")
                
                # excel_path = pdf_instance.pdf_file.url.replace(".pdf", ".xlsx")
                # output_pdf_path = pdf_instance.pdf_file.url.replace(".pdf", "_cost.pdf")
                session_data = UploadSession.objects.get(session_id=session_id)
                session_data.pdf_path = file_path
                session_data.output_pdf_path = output_pdf_path
                session_data.excel_path = excel_path
                session_data.calculated_price = cost*total_pages/100
                session_data.pdf_uploaded = True
                session_data.country = country
                session_data.currency = currency
                session_data.status = 'pdf_uploaded'
                session_data.project_title = project_title
                session_data.save()
                
                if not os.path.exists(directory_path):
                    return JsonResponse({"error": "PDF directory not found."}, status=400)
                
                session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price_data': {
                            'currency': 'usd',
                            'unit_amount': cost,  # amount in cents
                            'product_data': {
                                'name': 'AutoCAD AI Estimation'
                            },
                        },
                        'quantity': total_pages,
                    }],
                    mode='payment',
                    success_url=f'https://admin.globalpreservicesllc.com/autocad-ai-estimate?session_id={session_id}',
                    cancel_url='https://gps.globalpreservicesllc.com/',
                    metadata={'pdf_id': session_id}
                )

                # return JsonResponse({'checkout_url': session.url})
                #wix update
                # future = executor.submit(start_pdf_processing, file_path, excel_path, output_pdf_path, country, currency, session_id)
                
                # thread = threading.Thread(target=start_pdf_processing, args=(file_path, excel_path, output_pdf_path))
                # thread.start()
                
                return JsonResponse({
                "success": True,
                "message": "PDF uploaded successfully, You need to pay to proceed. Once the payment is completed you will be notified",
                "amount" : cost,
                "checkout_url": session.url,
                "file_url": pdf_instance.pdf_file.url,  # Public URL
                "file_path": pdf_instance.pdf_file.path  # Full file path on the server
                }, status=200)
    except Exception as e:
        return JsonResponse({"success" : False, "error": str(e)})    
    
def create_session(request):
    session_id = str(uuid4())
    # Save session to DB with status = "awaiting_upload"
    session=UploadSession.objects.create(session_id=session_id, status='awaiting_upload')
    return JsonResponse({"session_id": session_id})


@csrf_exempt
def stripe_webhook(request):
    payload = request.body
    sig_header = request.META['HTTP_STRIPE_SIGNATURE']
    endpoint_secret = 'whsec_IQzs9bd8N7r2HbXtoH1MpZTFVJEUvsdR'

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        return JsonResponse(status=400)
    except stripe.error.SignatureVerificationError:
        return JsonResponse(status=400)

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        pdf_id = session['metadata']['pdf_id']
        session_data = UploadSession.objects.get(session_id=pdf_id)
        file_path = session_data.pdf_path
        excel_path = session_data.excel_path
        output_pdf_path = session_data.output_pdf_path
        country = session_data.country
        currency = session_data.currency
        project_title = session_data.project_title
        session_data.payment_completed = True
        session_data.status = 'payment_completed'
        session_data.save()
        future = executor.submit(start_pdf_processing, file_path, excel_path, output_pdf_path, country, currency, pdf_id, project_title)
        # Resume processing
        # process_pdf_after_payment(pdf_id)

    return JsonResponse(status=200)

@csrf_exempt
def classify_csi_code(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        job_description = data.get('job_description', '')

        if not job_description:
            return JsonResponse({'error': 'Job description is required.'}, status=400)

        # GPT Prompt
        system_prompt = (
            "You are a construction estimator AI. Based on the CSI MasterFormat (Divisions 01ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“50), "
            "return the most appropriate CSI code and its description for the given job activity. "
            "Respond with JSON format like: {'code': '31 23 16.13', 'title': 'Trenching for Footings'}"
        )

        user_prompt = f"Job Activity: {job_description}"

        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",  # or gpt-3.5-turbo for cheaper
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            ai_reply = response.choices[0].message.content
            json_ai_reply = extract_json_from_response(ai_reply)
            print("AI Reply:", json_ai_reply)
            # Try to safely parse the response if it's valid JSON
            try:
                result = json.loads(json_ai_reply)
                code = result.get('code', '')
                title = result.get('title', '')
                return JsonResponse({
                'success': True,
                'code': code,
                'title': title,
                }, status=200)
           
            except json.JSONDecodeError:
                result = {'raw_response': ai_reply, 'error': 'Invalid JSON from GPT'}
                return JsonResponse(result, status=500)
            

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST allowed'}, status=405)
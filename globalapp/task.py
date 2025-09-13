import os
import threading
import time
import pandas as pd
from pdf2image import convert_from_path
import openai
from django.core.cache import cache
from django.conf import settings
import pytesseract
import cv2
import openpyxl
import re
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape, A3
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from django.http import JsonResponse
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


API_KEY = "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"
client = openai.OpenAI(api_key=API_KEY)
active_threads = []
lock = threading.Lock()
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
#pytesseract.pytesseract.tesseract_cmd = r"\var\Django\Tesseract-OCR\tesseract.exe"
table_data= [["CSI code", "Category", "Job Activity", "Quantity", "Unit", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]]
material_sum = 0
equipment_sum = 0
labor_sum = 0
total_sum = 0
row = []

def notify_frontend(pdf_url, excel_url):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "pdf_processing", 
        {
            "type": "notify_completion",
            "pdf_url": pdf_url,
            "excel_url": excel_url,
        }
    )
    
def extract_json_from_response(response_text):
    """Extracts JSON content from OpenAI's response."""
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1)  # Extract JSON part
    return None  # Handle case if no JSON is found

# def extract_text_from_cad(image_path):
#     print("entered in ocr", image_path)
#     try:
#         img = cv2.imread(image_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         text = pytesseract.image_to_string(gray)
#         print("ocr rext", text)
#     except Exception as e:
#         print("OCR Exception  {e}")
#     return text
def extract_text_from_cad(image_path):
    print(f"Entered OCR function: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found - {image_path}")
        return ""

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: cv2.imread failed - {image_path}")
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        print("OCR text output:", text[:200])  # Print first 200 characters for debugging
        return text

    except Exception as e:
        print(f"OCR Exception: {e}")
        return ""
def get_construction_jobs(cad_text):
    print("here is construction jobs")
    prompt = f"""
    Analyze the following CAD drawing and extract all relevant construction job categories and activities based on the latest MasterFormat standards to complete . Provide estimated material, labor, and equipment costs to complete the work in cad drawing in the USA.
    reference the lastest MasterFormat standards while analyzing the following CAD drawing.
    Cost must consider the quantities, sizes or volumes of each item.
    Unit must be short form like SF, CY, LF, EA, etc.
    No include any narrative like  // hypothetical cost, // example USD per SF, //example quantity etc.
    No include any description at the start or end in json response like // More categories and activities would follow here based on detailed project data, etc.
    {cad_text}
    
    Output should be formatted as a structured JSON with fields: CSI code, Category, Job Activity, Quantity,  Unit, Rate,  Material Cost, Equipment Cost, Labor Cost, Total Cost(Material Cost + Equipment Cost + Labor Cost).
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert in construction estimating."},
                  {"role": "user", "content": prompt}]
    )
    print("response===============>",response)  
    return response.choices[0].message.content

def check_json_format(data):
    """
    Check if JSON data is in the format:
    1. A list of dictionaries: [{ "category": "cat1", "cost": "4223" }, {...}, {...}]
    2. A dictionary with a single key mapping to a list of dictionaries: { "item": [{...}, {...}] }
    
    Returns:
        "list_format" if it's a list of dictionaries.
        "dict_format" if it's a dictionary containing a key with a list of dictionaries.
        "invalid_format" otherwise.
    """
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return "list_format", data

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return "dict_format", value  # Extract the list from the dictionary

    return "invalid_format", None

# Main processing function
def process_cad_drawing(image_path, output_excel, pdf_output_path):
    global material_sum, equipment_sum, labor_sum, total_sum, row
    print("image_path", image_path, output_excel, pdf_output_path)
    cad_text = extract_text_from_cad(image_path)
    job_data = get_construction_jobs(cad_text)
    
     # print("job_data===============>",job_data)
    # job_data_parsed = pd.read_json(job_data)  # Assuming OpenAI returns JSON
    # save_to_excel(job_data_parsed, output_excel)
    clean_json_str = extract_json_from_response(job_data)
    # cost_estimation_elements = clean_json_str[0]
    # print("cost_estimation_elements===============>", cost_estimation_elements)
    
    if clean_json_str:
        try:
            job_data_parsed = json.loads(clean_json_str)
            print("job_data_parsed===============>",job_data_parsed)
            # Convert JSON string to Python object
            data_type, data_list = check_json_format(job_data_parsed)
            # data_list = job_data_parsed
            
            # Convert JSON data to a list of lists (table data) including header row.
            if data_list:
                # headers = list(data_list[0].keys())
                # table_data = [headers]  # header row
                for item in data_list:
                    # row = [item.get(header, "") for header in headers]
                    rate = float(item.get("Total Cost"))/float(item.get("Quantity"))
                    fixed_rate = round(rate, 2)
                    formatted_rate = f"{fixed_rate:,.2f}"
                    quantity = int(item.get("Quantity", ""))
                    formatted_quantity = format(quantity, ",")
                    material_cost = round(float(item.get("Material Cost", 0)), 2)
                    formatted_material_cost = f"{material_cost:,.2f}"
                    equip_cost = round(float(item.get("Equipment Cost", 0)), 2)
                    formatted_equip_cost = f"{equip_cost:,.2f}"
                    labor_cost = round(float(item.get("Labor Cost", 0)), 2)
                    formatted_labor_cost = f"{labor_cost:,.2f}"
                    total_cost = round(float(item.get("Total Cost", 0)), 2)
                    formatted_total_cost = f"{total_cost:,.2f}"
                    with lock:
                        row = [item.get("CSI code"), item.get("Category", ""), item.get("Job Activity", ""), formatted_quantity, item.get("Unit", ""), f'${formatted_rate}',
                          f'${formatted_material_cost}', f'${formatted_equip_cost}', f'${formatted_labor_cost}', f'${formatted_total_cost}']
                    # table_data.append(row)
                    # material_sum += item.get("Material Cost", 0)
                    # equipment_sum += item.get("Equipment Cost", 0)
                    # labor_sum += item.get("Labor Cost", 0)
                    # total_sum += item.get("Total Cost", 0)
                    
                        table_data.append(row)
                        material_sum += float(item.get("Material Cost", 0))
                        equipment_sum += float(item.get("Equipment Cost", 0))
                        labor_sum += float(item.get("Labor Cost", 0))
                        total_sum += float(item.get("Total Cost", 0))
            # else:
            #     table_data = [["No Data"]]
            # material_sum = round(material_sum, 2)
            # equipment_sum = round(equipment_sum, 2)
            # labor_sum = round(labor_sum, 2)
            # total_sum = round(total_sum, 2)
            
            # Define the PDF output file name
            # pdf_file = "construction_jobs.pdf"
            
            # Create a PDF document with landscape orientation
                
        except json.JSONDecodeError as e:
            print("Error parsing JSON response:", e)
    else:
        print("No valid JSON found in the response.")

def process_pdf_page(pdf_path, page_number, output_excel, output_pdf):
    try:
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=500, poppler_path="/usr/bin")
        
        if not images:
            return
        image = images[0]
        directory_path = os.path.dirname(pdf_path)
        image_path = f"{directory_path}/page_{page_number}.png"
        image.save(image_path, "PNG")
        print("image_path===>", image_path)
        process_cad_drawing(image_path, output_excel, output_pdf)
        
    except Exception as e:
        print(f"Error processing {pdf_path}, page {page_number}: {e} ")
    finally:
        with lock:
            active_threads.remove(threading.current_thread())
        if not active_threads:
            
            print("All PDF page processing threads have completed.")
            formatted_material_sum = f"{material_sum:,.2f}"
            formatted_equipment_sum = f"{equipment_sum:,.2f}"
            formatted_labor_sum = f"{labor_sum:,.2f}"
            formatted_total_sum = f"{total_sum:,.2f}"
            table_data.append(["","", "", "", "", "Total", f'${formatted_material_sum}', f'${formatted_equipment_sum}', f'${formatted_labor_sum}', f'${formatted_total_sum}'])
            doc = SimpleDocTemplate(output_pdf, pagesize=landscape(A3))
            table = Table(table_data, repeatRows=1)

                # Add some basic style to the table
            style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    # Special style for the last row ("Total" row)
                    ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),  # Light grey background
                    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),  # Bold text
                    ('TEXTCOLOR', (0, -1), (-1, -1), colors.black),  # Black text
                ])
            table.setStyle(style)

            # Build the PDF document
            doc.build([table])

            print(f"PDF generated and saved as '{output_pdf}'.")
            # Create an Excel workbook and sheet
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Cost Estimation"

            # Write data to the sheet
            for row in table_data:
                ws.append(row)

            # Save the Excel file
            wb.save(output_excel)
            print(f"Data saved to {output_excel}")
            base_path = r"/var/Django/cadProcessor/media"
            pdf_url=os.path.relpath(output_pdf, base_path).replace("\\", "/")
            excel_url = os.path.relpath(output_excel, base_path).replace("\\", "/")
            notify_frontend(pdf_url, excel_url)
            # return JsonResponse({"success": True, "message": "successfully generated estimation files"})
def start_pdf_processing(pdf_file, output_excel, output_pdf):
    global active_threads
    total_pages = len(convert_from_path(pdf_file, dpi=500, poppler_path="/usr/bin"))
    for page_number in range(1, total_pages+1):
        print("thread number ===>", page_number)
        thread = threading.Thread(target=process_pdf_page, args=(pdf_file, page_number, output_excel, output_pdf))
        active_threads.append(thread)
        thread.start()

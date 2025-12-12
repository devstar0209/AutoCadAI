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
from reportlab.lib.pagesizes import letter, landscape, A3, A2
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageTemplate, Frame, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from django.http import JsonResponse
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict
from django.db import connection
import fitz
import base64
image_index=1
API_KEY = ""
client = openai.OpenAI(api_key=API_KEY)
active_threads = []
lock = threading.Lock()
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
table_data= [["CSI code", "Category", "Job Activity", "Quantity", "Unit", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]]
pdf_total_pages = 0
row = []
table_elements = []
project_title=""
drawing_date=""
page_count = -1

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def split_image(image, dir_path):
    global image_index
    width, height = image.size
    part_width = width // 3
    part_height = height // 3

    parts = []

    for row in range(3):
        for col in range(3):
            left = col * part_width
            upper = row * part_height
            right = left + part_width
            lower = upper + part_height
            
            cropped_part = image.crop((left, upper, right, lower))  # Crop the region
            parts.append(cropped_part)

            # Save each part
            part_filename = os.path.join(dir_path, f"page_{image_index}.jpg")
            cropped_part.save(part_filename)
            print(f"Saved {part_filename}")
            
            image_index += 1  # Increment sequence

def split_pdf_by_size(pdf_path, max_size_mb=5):
    """Splits a PDF into multiple parts, each = max_size_mb MB."""
    doc = fitz.open(pdf_path)
    output_files = []
    part_num = 1
    current_doc = fitz.open()
    temp_path = f"split_part_{part_num}.pdf"

    for page_num in range(len(doc)):
        current_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        current_doc.save(temp_path)

        # Check file size
        if os.path.getsize(temp_path) > max_size_mb * 1024 * 1024:
            # Save the previous file as a finalized split
            final_part_path = f"split_part_{part_num}.pdf"
            current_doc.save(final_part_path)
            output_files.append(final_part_path)

            # Start a new PDF
            part_num += 1
            current_doc = fitz.open()
            temp_path = f"split_part_{part_num}.pdf"

    # Save the last split if it exists
    if len(current_doc) > 0:
        final_part_path = f"split_part_{part_num}.pdf"
        current_doc.save(final_part_path)
        output_files.append(final_part_path)

    current_doc.close()
    return output_files

def process_pdf_splits(directory, pdf_path):
    """Splits a PDF and converts each split part into images."""
    global image_index
    split_files = split_pdf_by_size(pdf_path, max_size_mb=5)

    for split_pdf in split_files:
        print(f"Processing: {split_pdf}")
        images = convert_from_path(split_pdf, dpi=300)  # Adjust DPI as needed

        for i, img in enumerate(images):
            # img.save(f"{directory}\page_{image_index}.png", "PNG")  # Save images
            # image_index +=1
            split_image(img, directory)
        os.remove(split_pdf)  # Cleanup after processing

def initialize_globals():
    global table_data, pdf_total_pages, project_title, drawing_date, page_count, row, image_index
    table_data = [["CSI code", "Category", "Job Activity", "Quantity", "Unit", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]]
    pdf_total_pages = 0
    project_title = ""
    drawing_date = ""
    page_count = -1
    row = []
    image_index = 1
def notify_frontend(total_page, cur_page, message, pdf_url, excel_url, session_id):
    global page_count
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"pdf_processing_{session_id}", 
        {
            "type": "notify_completion",
            "message" : message,
            "total_page" : total_page,
            "cur_page" : cur_page,
            "pdf_url": pdf_url,
            "excel_url": excel_url,
        }
    )
def summarize_costs_by_division(cost_data, currency):
    summary = defaultdict(lambda: [0, 0, 0, 0])  # Stores [Material Cost, Equipment Cost, Labor Cost, Total Cost]
    
    # Mapping of divisions (you can expand this list as needed)
    division_names = {
    "01": "General Requirements",
    "02": "Existing Conditions",
    "03": "Concrete",
    "04": "Masonry",
    "05": "Metals",
    "06": "Wood, Plastics, and Composites",
    "07": "Thermal and Moisture Protection",
    "08": "Openings",
    "09": "Finishes",
    "10": "Specialties",
    "11": "Equipment",
    "12": "Furnishings",
    "13": "Special Construction",
    "14": "Conveying Equipment",
    "21": "Fire Suppression",
    "22": "Plumbing",
    "23": "HVAC (Heating, Ventilating, and Air Conditioning)",
    "25": "Integrated Automation",
    "26": "Electrical",
    "27": "Communications",
    "28": "Electronic Safety and Security",
    "31": "Earthwork",
    "32": "Exterior Improvements",
    "33": "Utilities",
    "34": "Transportation",
    "35": "Waterway and Marine Construction",
    "40": "Process Integration",
    "41": "Material Processing and Handling Equipment",
    "42": "Process Heating, Cooling, and Drying Equipment",
    "43": "Process Gas and Liquid Handling, Purification, and Storage Equipment",
    "44": "Pollution Control Equipment",
    "45": "Industry-Specific Manufacturing Equipment",
    "46": "Water and Wastewater Equipment",
    "48": "Electrical Power Generation"
    }

    
    for row in cost_data[1:]:  # Skip header row
        csi_code = row[0]
        division = csi_code.split(" ")[0][:2]  # Extract the first two digits
        
        material_cost = float(row[6].replace(f"{currency}$", "").replace(",", ""))
        equipment_cost = float(row[7].replace(f"{currency}$", "").replace(",", ""))
        labor_cost = float(row[8].replace(f"{currency}$", "").replace(",", ""))
        total_cost = float(row[9].replace(f"{currency}$", "").replace(",", ""))
        
        summary[division][0] += material_cost
        summary[division][1] += equipment_cost
        summary[division][2] += labor_cost
        summary[division][3] += total_cost
    
    # Convert summary to list format
    result = []
    for division, costs in summary.items():
        division_name = division_names.get(division, "Unknown Division")
        formatted_material_cost = f"{costs[0]:,.2f}"
        formatted_equip_cost = f"{costs[1]:,.2f}"
        formatted_labor_cost = f"{costs[2]:,.2f}"
        formatted_total_cost = f"{costs[3]:,.2f}"
        result.append([f"Division {division}", division_name, f"{currency}${formatted_material_cost}", f"{currency}${formatted_equip_cost}", f"{currency}${formatted_labor_cost}", f"{currency}${formatted_total_cost}"])
    summary_material_sum = sum(float(row1[2].replace(f"{currency}$", "").replace(",", "")) for row1 in result[0:])
    summary_equipment_sum = sum(float(row1[3].replace(f"{currency}$", "").replace(",", "")) for row1 in result[0:])
    summary_labor_sum = sum(float(row1[4].replace(f"{currency}$", "").replace(",", "")) for row1 in result[0:])
    summary_total_sum = sum(float(row1[5].replace(f"{currency}$", "").replace(",", "")) for row1 in result[0:])
    formatted_summary_material_sum = f"{summary_material_sum:,.2f}"
    formatted_summary_equipment_sum = f"{summary_equipment_sum:,.2f}"
    formatted_summary_labor_sum = f"{summary_labor_sum:,.2f}"
    formatted_summary_total_sum = f"{summary_total_sum:,.2f}"
    result.append(["", "Total", f'{currency}${formatted_summary_material_sum}', f'{currency}${formatted_summary_equipment_sum}', f'{currency}${formatted_summary_labor_sum}', f'{currency}${formatted_summary_total_sum}'])
    return result    
def extract_json_from_response(response_text):
    """Extracts JSON content from OpenAI's response."""
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1)  # Extract JSON part
    return None  # Handle case if no JSON is found

def extract_text_from_cad(image_path):
    img = Image.open(image_path).convert("L")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(img)
    return text
def get_title_date(cad_text):
    prompt = f"""
    Analyze this CAD drawing and extract drawing date.
    date should be in the format of MM/DD/YYYY.
    {cad_text}
    Output should be formatted as a structured JSON with fields: date.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert in construction estimating."},
                  {"role": "user", "content": prompt}]
    )
    print("response===============>",response)  
    return response.choices[0].message.content

def getQuantityTakeoff(base64):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Provide a take-off of the  quantities using the scale from this engineering drawing  returning as a json formart"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content
def get_construction_jobs(quantity_data, country):
    print("quantity data", quantity_data)
    prompt = f"""
    You are an AI trained in construction cost estimation in the {country} using the CSI MasterFormat. Based on the following site plan quantities, generate a job activity list including: 
    Site Quantities:
    {quantity_data} .
    reference the lastest MasterFormat standards.
    Cost must consider the quantities, sizes or volumes of each item.
    Unit must be short form like SF, CY, LF, EA, etc.
    Material cost for Earth work should be 0.
    No include any narrative like  // hypothetical cost, // example USD per SF, //example quantity etc.
    No include any description at the start or end in json response like // More categories and activities would follow here based on detailed project data, etc.
    Output should be formatted as a structured JSON with fields: CSI code, Category, Job Activity, Quantity,  Unit, Local Currency, Rate,  Material Cost, Equipment Cost, Labor Cost, Total Cost(Material Cost + Equipment Cost + Labor Cost).
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

def header_footer_text(image_path):
    cad_text = extract_text_from_cad(image_path)
        
    job_data = get_title_date(cad_text)
        
        # print("job_data===============>",job_data)
        # job_data_parsed = pd.read_json(job_data)  # Assuming OpenAI returns JSON
        # save_to_excel(job_data_parsed, output_excel)
    clean_json_str = extract_json_from_response(job_data)
        # cost_estimation_elements = clean_json_str[0]
        # print("cost_estimation_elements===============>", cost_estimation_elements)
        
    if clean_json_str:
            
                job_data_parsed = json.loads(clean_json_str)
                print("job_data_parsed===============>",job_data_parsed)
                return job_data_parsed
    else:
        no_response = {'date': ''}
        return no_response
# Main processing function
def process_cad_drawing(total_pages, page_number, image_path, output_excel, pdf_output_path, country, currency, session_id):
    global row, table_data, project_title, drawing_date, page_count
    if page_number <= 9:
        cad_info=header_footer_text(image_path)
        drawing_date = cad_info.get("date", "")
    print("image_path", image_path, output_excel, pdf_output_path)
    # cad_text = extract_text_from_cad(image_path)
    base64_image = encode_image(image_path)
    job_data = getQuantityTakeoff(base64_image)
    quantity_list = extract_json_from_response(job_data)
    print("quantity_list", quantity_list)
    job_activities = get_construction_jobs(quantity_list, country)
     # print("job_data===============>",job_data)
    # job_data_parsed = pd.read_json(job_data)  # Assuming OpenAI returns JSON
    # save_to_excel(job_data_parsed, output_excel)
    clean_json_str = extract_json_from_response(job_activities)
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
                    # currency = item.get("Local Currency", "")
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
                        csi_code = item.get("CSI code").strip()
                        if csi_code=="":
                            csi_code = "00 00 00"
                        row = [csi_code, item.get("Category", ""), item.get("Job Activity", ""), formatted_quantity, item.get("Unit", ""), f'{currency}${formatted_rate}',
                          f'{currency}${formatted_material_cost}', f'{currency}${formatted_equip_cost}', f'{currency}${formatted_labor_cost}', f'{currency}${formatted_total_cost}']
                    # table_data.append(row)
                    # material_sum += item.get("Material Cost", 0)
                    # equipment_sum += item.get("Equipment Cost", 0)
                    # labor_sum += item.get("Labor Cost", 0)
                    # total_sum += item.get("Total Cost", 0)
                    
                        table_data.append(row)
                        
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
    page_count += 1
    notify_frontend(total_pages, page_count, "Processing is in progress...", "", "", session_id) 
def process_pdf_page(total_pages, pdf_path, page_number, output_excel, output_pdf, image_path, country, currency, session_id):
    try:
        
        process_cad_drawing(total_pages, page_number, image_path, output_excel, output_pdf, country, currency, session_id)
        
    except Exception as e:
        print(f"Error processing {pdf_path}, page {page_number}: {e} ")

def remove_duplicates(data, currency):
    seen = set()
    unique_data = []
    
    for item in data:
        category = item[0]  # Extract Category & Job Activity
        key = (category)
        
        if key not in seen:
            seen.add(key)
            unique_data.append(item)

    return unique_data

def header_footer(canvas, doc, project_title, pdf_total_pages):
    """
    Custom function to add header and footer on each page.
    """
    global  drawing_date
    page_width, page_height = landscape(A2)
    table_width = page_width * 0.90  # 90% of A3 width
    styles = getSampleStyleSheet()
    try:
        current_page = canvas.getPageNumber()  # Get current page number
        print("Current Page:", current_page)
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=12, spaceAfter=10)
        date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], alignment=2, fontSize=12)
        title = Paragraph(f"<b>{project_title}</b>", title_style)
        date = Paragraph(f"<b>{'Estimate Published Date: '+ datetime.today().strftime('%m/%d/%Y')}</b>", date_style)
        # Create a table for alignment
        title_date_table = Table([[title, date]], colWidths=[table_width * 0.5, table_width * 0.5])
        title_date_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        if current_page == pdf_total_pages:
            summary_title = "Project Cost Summary"
            title = Paragraph(f"<b>{project_title}</b><br/><b>{summary_title}</b>", title_style)
            title_date_table = Table([[title, date]], colWidths=[table_width * 0.5, table_width * 0.5])
            title_date_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
            
        # Header: Project Title (Left) and Date (Right)
        
        

        # Draw the header
        width, height = landscape(A2)
        title_date_table.wrapOn(canvas, width, height)
        if current_page == pdf_total_pages:
            title_date_table.drawOn(canvas, width * 0.05, height - 80)  # Positioning at the top (5% margin)
        else:
            title_date_table.drawOn(canvas, width * 0.05, height - 50)
        footer_date_style = ParagraphStyle('FooterDateStyle', parent=date_style, alignment=TA_LEFT)
        if drawing_date:
            footer_date = Paragraph(f"<b>{'Drawing  Published Date: ' + str(drawing_date)}</b>", footer_date_style)
        else:
            footer_date = Paragraph(f"<b>{'Drawing  Published Date: ' + datetime.today().strftime('%m/%d/%Y')}</b>", footer_date_style)
        footer_date.wrapOn(canvas, page_width*0.90, page_height)
        footer_date.drawOn(canvas, page_width*0.05, 40)  # Adjusted Y position
    except Exception as e:
        print("Error in header_footer:", e)
def on_page(canvas, doc):
        global project_title, pdf_total_pages
        """Apply headers and footers with correct page numbering"""
        header_footer(canvas, doc, project_title, pdf_total_pages)
def count_pages(canvas, doc):
        global pdf_total_pages
        """Counts the total number of pages"""
        pdf_total_pages = canvas.getPageNumber()
        print("Total Pages=>>", pdf_total_pages)   
def start_pdf_processing(pdf_file, output_excel, output_pdf, country, currency, session_id, cad_title):
    global table_data, table_elements, page_count, pdf_total_pages, project_title, image_index
    dummy_elements = []
    project_title = cad_title
    print("poppler will act soon===>")
    directory_path = os.path.dirname(pdf_file)
    print("pdf file path==>", pdf_file)
    print("directory path", directory_path)
    # doc = fitz.open(pdf_file)
    # total_pages = len(doc)
    # print("page_count===>",total_pages)
    
    process_pdf_splits(directory_path, pdf_file)
    total_pages = image_index
    # images = convert_from_path(pdf_file, dpi=300, poppler_path=r"e:\Fine_tune_proj\Python\AutoCAD\poppler\Library\bin")
    
    # for page_number in range(total_pages):
    #     page_number +=1
    #     images = convert_from_path(pdf_file, dpi=300, first_page=page_number, last_page=page_number, poppler_path=r"e:\Fine_tune_proj\Python\AutoCAD\poppler\Library\bin")
    #     image_path = f"{directory_path}\page_{page_number}.png"
    #     images[0].save(image_path, "PNG")

    
    with ThreadPoolExecutor(max_workers=min(2, total_pages)) as executor:
        futures = []
        for page_number in range(total_pages):
            page_number +=1
            image_path = f"{directory_path}/page_{page_number}.jpg"
            future = executor.submit(process_pdf_page, total_pages, pdf_file, page_number, output_excel, output_pdf, image_path, country, currency, session_id)
            futures.append(future)

        # Wait for all tasks to finish
        for future in futures:
            future.result()

    # Notify frontend when all processing is done
    print("All PDF page processing threads have completed.")
    
    page_width, page_height = landscape(A2)
    table_width = page_width * 0.90
    
    col_widths = [table_width / len(table_data[0])] * len(table_data[0]) 
    
    header = table_data[0]
    sorted_data = sorted(table_data[1:], key=lambda x: x[0])
    cleaned_data = remove_duplicates(sorted_data, currency)
    final_data = [header] + cleaned_data
    material_sum = 0
    equipment_sum = 0
    labor_sum = 0
    total_sum = 0
    material_sum = sum(float(row1[6].replace(f"{currency}$", "").replace(",", "")) for row1 in final_data[1:])
    equipment_sum = sum(float(row1[7].replace(f"{currency}$", "").replace(",", "")) for row1 in final_data[1:])
    labor_sum = sum(float(row1[8].replace(f"{currency}$", "").replace(",", "")) for row1 in final_data[1:])
    total_sum = sum(float(row1[9].replace(f"{currency}$", "").replace(",", "")) for row1 in final_data[1:]) # Total cost = Material + Equipment + Labor
    formatted_material_sum = f"{material_sum:,.2f}"
    formatted_equipment_sum = f"{equipment_sum:,.2f}"
    formatted_labor_sum = f"{labor_sum:,.2f}"
    formatted_total_sum = f"{total_sum:,.2f}"
    summary_header = ["Division", "Division Name", "Material Cost", "Labor Cost", "Equipment Cost", "Total Cost"]
    summary_result =[summary_header] + summarize_costs_by_division(final_data, currency)
    
    final_data.append(["","", "", "", "", "Total", f'{currency}${formatted_material_sum}', f'{currency}${formatted_equipment_sum}', f'{currency}${formatted_labor_sum}', f'{currency}${formatted_total_sum}'])
    frame = Frame(page_width * 0.05,  # Left margin
                  60,                 # Bottom margin (leave space for footer)
                  table_width,        # Content width
                  page_height - 120,  # Content height (leave space for header & footer)
                  id='main_frame')
    template = PageTemplate(id='custom', frames=[frame], onPage=lambda canvas, doc: header_footer(canvas, doc, project_title, pdf_total_pages))
    
    table = Table(final_data, repeatRows=1)
    
    style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (2, -1), 'LEFT'),
                    ('ALIGN', (3, 0), (-1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),  # Light grey background
                    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),  # Bold text
                    ('TEXTCOLOR', (0, -1), (-1, -1), colors.black),  # Black text
                ])
    table.setStyle(style)
    
    summary_table = Table(summary_result, repeatRows=1)
    summary_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (1, -1), 'LEFT'),
                    ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),  # Light grey background
                    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),  # Bold text
                    ('TEXTCOLOR', (0, -1), (-1, -1), colors.black),  # Black text
                ])
    summary_table.setStyle(summary_style)
    table_elements.append(table)
    table_elements.append(PageBreak())
    table_elements.append(summary_table)
    dummy_elements.append(table)
    dummy_elements.append(PageBreak())
    dummy_elements.append(summary_table)
    dummy_doc = SimpleDocTemplate("dummy.pdf", pagesize=landscape(A2))
    dummy_doc.addPageTemplates([template])
    dummy_doc.build(dummy_elements, onLaterPages=lambda canvas, doc: count_pages(canvas, doc), onFirstPage=lambda canvas, doc: count_pages(canvas, doc))
    print("pdf total pages=======>", pdf_total_pages)
    doc = SimpleDocTemplate(output_pdf, pagesize=landscape(A2))
    doc.addPageTemplates([template])
    
    doc.build(table_elements, onFirstPage=lambda canvas, doc: header_footer(canvas, doc, project_title, pdf_total_pages), onLaterPages=lambda canvas, doc: header_footer(canvas, doc, project_title, pdf_total_pages))

    print(f"PDF generated and saved as '{output_pdf}'.")
    final_data.extend(summary_result)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Cost Estimation"

    for row in final_data:
                ws.append(row)

    wb.save(output_excel)
    print(f"Data saved to {output_excel}")
    base_path = r"/var/Django/cadProcessor/media"
    pdf_url=os.path.relpath(output_pdf, base_path).replace("\\", "/")
    excel_url = os.path.relpath(output_excel, base_path).replace("\\", "/")
    initialize_globals()
    notify_frontend(total_pages, page_count, "Processing was completed", pdf_url, excel_url, session_id)
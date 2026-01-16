import os
import re
import json
import cv2
import pytesseract
from pytesseract import Output
import openai
import openpyxl
from openpyxl import Workbook
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape, A3, A2
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageTemplate, Frame, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from datetime import datetime
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import base64
from typing import List, Dict
import tiktoken

# =================== CONFIG ===================
API_KEY = ""
client = openai.OpenAI(api_key=API_KEY)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Linux
# pytesseract.pytesseract.tesseract_cmd = r"C:\Path\To\tesseract.exe"  # Windows
MAX_WORKERS = 4
MAX_TOKENS_PER_CHUNK = 8000
MODEL = "gpt-4o-2024-08-06" # "ft:gpt-4o-2024-08-06:global-precisional-services-llc::CSwogr3u"
table_elements = []
drawing_date=""
pdf_total_pages = 0

allowed_categories = [
    "General Requirements",
    "Existing Conditions",
    "Concrete",
    "Masonry",
    "Metal",
    "Wood, Plastics, and Composites",
    "Thermal and Moisture Protection",
    "Openings",
    "Finishes",
    "Specialties",
    "Equipment",
    "Furnishing",
    "Special Construction",
    "Conveying Systems",
    "Fire Suppression",
    "Plumbing",
    "HVAC",
    "Electrical",
    "Communications",
    "Electronic Safety & Security",
    "Earthwork",
    "Exterior Improvement",
    "Utilities",
    "Transportations",
    "Waterway & marine",
    "Material Processing & Handling Equipment",
    "Pollution Control Equipment"
]

# =================== FRONTEND NOTIFY ===================
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

# =================== PDF PAGE PROCESSING ===================
def convert_pdf_page_to_image(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(pdf_path, dpi=200, first_page=page_number, last_page=page_number, poppler_path="/usr/bin")
    if not images: return ""
    directory = os.path.dirname(pdf_path)
    image_path = os.path.join(directory, f"page_{page_number}.png")
    images[0].save(image_path, "PNG")
    return image_path

# =================== OCR ===================
def extract_text_from_image(image_path: str) -> str:
    print(f"Entered OCR function: {image_path}")
    min_confidence = 40
    line_gap = 15
    try:
        if not os.path.exists(image_path):
            return ""

        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Failed to load image: {image_path}")
            return ""

        # === Preprocess image ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)  # edge-preserving denoise
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        thresh = cv2.adaptiveThreshold(enhanced, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # === OCR with position data ===
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

        lines = []
        current_line = []
        last_y = None

        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            conf = int(data["conf"][i])
            if conf < min_confidence:
                continue

            y = data["top"][i]
            if last_y is None or abs(y - last_y) < line_gap:
                current_line.append(word)
            else:
                # new line detected
                lines.append(" ".join(current_line))
                current_line = [word]
            last_y = y

        if current_line:
            lines.append(" ".join(current_line))

        # === Post-cleaning ===
        cleaned_lines = []
        for line in lines:
            # print(f"line:: {line}")
            # remove single-character noise
            if len(line.strip()) < 2:
                continue
            # collapse extra spaces
            line = " ".join(line.split())
            # line = re.sub(r'(M\.?H\.?)[.,]?', 'manhole', line, flags=re.IGNORECASE)
            cleaned_lines.append(line)

        structured_text = "\n".join(cleaned_lines)
        return structured_text.strip()

    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return ""

# =================== AI COST ESTIMATION ===================

def get_construction_jobs(cad_text, category, project_title, project_location, unit, prompt):
    print(f"Starting construction category {category} jobs analysis...")
    print(f"Custome prompt:: {prompt}")

    system_prompt = """
You are an expert construction take-off extractor and material cost and labor/equipment rate provider.
Refer local market benchmark corresponding to project location.
Your job is to extract job activities from the OCR text of CAD drawings and provide detailed quantity take-offs and Material cost per Unit and Labor/Equipment rates in local market benchmark corresponding to project location.
Only use the information explicitly present in the OCR text provided.
"""


    user_prompt = f"""
{prompt}
Please provide detailed construction take-off activities, quantities, material cost and labor/equipment rates from the following
OCR text:
{cad_text}.

Location: {project_location}.
Project Type: {project_title}.
Apply {unit} units.
Allow categories in {', '.join(allowed_categories)}.

Consider following rules for each item in your estimate:
- CONVERT text dimensions from OCR text If Imperal units is selected. (for example, 50 or 50 mm convert to 2" ; 150 or 150 mm convert to 6"; 100 or 100 mm concert to 4" ; 50x150 convert to 2x6; 50x200 convert to 2x8; 12mm to 0.5"). If Metric units is selected, convert text dimensions to Metric. ( for example 2 or 2" convert to 50mm; 4" convert to 100 mm; 6" convert to 150mm ; 8" convert to 200mm; 2x6 convert to 50x150; 2x8 convert to 50x200)
- M.UCost = material cost PER UNIT ONLY.
- L.Rate = labor cost PER HOUR ONLY.
- E.Rate = equipment cost PER HOUR ONLY.
- Skip exact duplicate description.
- Use 2024 MasterFormat (CSI) codes.
- The assigned Category MUST match the CSI Division.
- Apply net quantity measurement principles (measure to structural faces, exclude waste unless specified. based on converted dimensions)
- If OCR text includes Pitch Roof, Hip Roof, Slope Roof, Lean-to-Roof, Flat Roof, Mansard Roof, Open Gable End Roof, Dome Roof, Butterfly Roof, A-Farme Roof, Pyramid Roof, Gambrel Roof, Dutch Gable Roof, Bonnet Roof, A-Frame Roof, you must add Roof felt, roof covering ( like..Decking board, Roof Sheathing, Asphalt Shingle, Torch down membrain, wood shingle, Aluminum Standing Seam, Sheeting, Metal Tile Roofing, Clay Tile Roofing, and the like) and 2*2 lath, 2*4 Wall Plate, 2*6 Rafters, 2*8 Hip Rafter, 2*8 Valley Rafters, 2*10 Ridge Board, 2*10 Fascia Board, 1*12 Notched Blocking Board to roof eave. You MUST also check for exact related roof Slope/ Pitch and roofing scope present.
- If OCR text includes MH#1, MH#2, etc. You MUST check for exact manhole quantity, size, depth, and inter level, including related  manhole scope present.
- Currency must be native currency for the location (e.g., JMD for Jamaica, BBD for Barbados, etc.)
- If Location is USA or non-Commonwealth country, adjust only RSMeans cost standards. Electrical labor follows NECA 2023-2024.
- If Location is in the Caribbean or any Commonwealth country, adjust local market benchmarks and ONLY RICS/NRM2 measurement.

Return a valid JSON array of objects, one per item, with these fields:
- CSI Code (format 01 02 03.04)
- Category
- Job Activity
- Quantity (round number, use default only if missing)
- Unit
- M.UCost
- L.Rate
- L.Hrs (round number)
- E.Rate
- E.Hrs (round number)

Validation:
- NEVER change the CSI code to fit the Category.
- CSI code MUST be no empty.
- Category MUST be no empty.
- Unit can't be "wk".
- If unit is imperial unit, Unit is SF for concrete paving, CY for other concrete works. If metric unit, Base concrete works measured in cubic metres (m3) and M.UCost MUST be defined per cubic metre (m3),and other concrete works measured in square metres (m2) and the M.UCost MUST be recalculated using slab thickness (M.UCost (m2) = M.UCost (m3) * slab thickness (in metres)).
- Output units must be EXCLUSIVELY {unit}. Do not include the other measurement system.
- Unit is a DERIVED FIELD. Do not infer units semantically.
- Output job activity must include EXCLUSIVELY {unit}. Do not include the other measurement system.
- You MUST produce accurate market rate and cost for labor, equipment, material for {project_location} for cost estimate.
- Material cost for Only Earthwork should be 0.
- If conversion is applied, the resulting M.UCost MUST be lower than the m3 rate.
- If m2 and m3 M.UCost are equal, output is INVALID.
- When selected Imperial unit of measurement, all dimensions in Job Activity MUST be in imperial units (inches, feet, etc). Anything else, it's INVALID.
- When selected Metric unit of measurement, all dimensions in Job Activity MUST be in metric units (m, mm, etc). Anything else, it's INVALID.
- Return only valid JSON array, no extra text.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            store= True,
        )
        response_text = response.choices[0].message.content

        # Remove markdown fences and extra whitespace
        response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.DOTALL).strip()
        response_text = response_text.replace('```', '').strip()
        # print(f"AI response received:: {response_text}")
        
        return response_text
    except Exception as e:
        print(f"Error in get_construction_jobs: {e}")
        return ""

# =================== OUTPUT GENERATION ===================
def generate_outputs(output_json: dict, project_title: str, currency: str, output_excel: str, output_pdf: str):
    print(f"Generating Project {project_title} outputs...")
    """
    Save structured JSON estimate into Excel with Summary + Details.
    """
    summary_table_data = []
    detail_table_data = []

    wb = Workbook()

    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    ws_summary = wb.active
    ws_summary.title = "Summary"

    ws_summary.merge_cells('A1:O1')

    title_cell = ws_summary['A1']
    title_cell.value = project_title

    title_cell.font = Font(
        size=16,
        bold=True
    )

    title_cell.alignment = Alignment(
        horizontal='center',
        vertical='center'
    )
    ws_summary.row_dimensions[1].height = 40

    # Define styles
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    summary_font = Font(bold=True)
    summary_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
    border_style = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    summary_table_data.append(["Div", "Description", "Total"])  # headers    
    ws_summary.append(["Div", "Description", "Total"]) # headers

    # Style header
    for col in range(1, 4):
        cell = ws_summary.cell(row=2, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border_style
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Set column widths
    ws_summary.column_dimensions['A'].width = 5
    ws_summary.column_dimensions['B'].width = 50
    ws_summary.column_dimensions['C'].width = 18

    for item in output_json.get("Summary", []):
        summary_table_data.append([
            item.get("Div", 0),
            item.get("Category", ""),
            currency+"$"+str(format(item.get("Total Cost", 0), ","))
        ])
        ws_summary.append([
            item.get("Div", 0),
            item.get("Category", ""),
            currency+"$"+str(format(item.get("Total Cost", 0), ","))
        ])

    
    # --- Write Details Sheet ---
    ws_details = wb.create_sheet("Details")
    headers = ["Div", "CSI Code", "Description", "Quant.", "Unit", "M.U/Cost", "Total.Material", "Equip.Hrs","Equip.Rate", "Total.Equip","Labor.Hrs","Labor.Rate", "Total.Labor", "Sub Markups", "Subtotal Cost"]
    detail_table_data.append(headers)
    ws_details.append(headers)

    # Style header
    for col in range(1, 16):
        cell = ws_details.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border_style
        cell.alignment = Alignment(horizontal='center', vertical='center')

    # Column widths
    ws_details.column_dimensions['A'].width = 6
    ws_details.column_dimensions['B'].width = 12
    ws_details.column_dimensions['C'].width = 60
    for col in ['D']:
        ws_details.column_dimensions[col].width = 10
    for col in ['F','G','H','I','J','K','L','M','N','O','P','Q']:
        ws_details.column_dimensions[col].width = 10

    for item in output_json.get("Details", []):
        
        ucost = round(item.get("M.UCost", 0), 2)
        qty = round(item.get("Quantity", 0), 2)
        lrate = round(item.get("L.Rate", 0), 2)
        lhrs = item.get("L.Hrs", 0)
        erate = round(item.get("E.Rate", 0), 2)
        ehrs = item.get("E.Hrs", 0)
        mcost = round(ucost * qty, 2)
        lcost = round(lrate * lhrs, 2)
        ecost = round(erate * ehrs, 2)
        sub_markups = round((mcost + lcost + ecost) * 0.25, 2)
        subtotal = format(round(mcost + lcost + ecost + sub_markups, 2), ",")

        ws_details.append([
            item.get("Div", 0),
            item.get("CSI Code", ""),
            item.get("Job Activity", ""),
            format(qty, ","),
            item.get("Unit", "").lower(),
            currency+"$"+str(format(ucost, ",")),
            currency+"$"+format(mcost, ","),
            ehrs,
            currency+"$"+str(format(erate, ",")),
            currency+"$"+format(ecost, ","),
            lhrs,
            currency+"$"+str(format(lrate, ",")),
            currency+"$"+format(lcost, ","),
            currency+"$"+format(sub_markups, ","),
            currency+"$"+str(subtotal)
        ])
        detail_table_data.append([
            item.get("Div", 0),
            item.get("CSI Code", ""),
            item.get("Job Activity", ""),
            format(qty, ","),
            item.get("Unit", "").lower(),
            currency+"$"+str(format(ucost, ",")),
            currency+"$"+format(mcost, ","),
            ehrs,
            currency+"$"+str(format(erate, ",")),
            currency+"$"+format(ecost, ","),
            lhrs,
            currency+"$"+str(format(lrate, ",")),
            currency+"$"+format(lcost, ","),
            currency+"$"+format(sub_markups, ","),
            currency+"$"+str(subtotal)
        ])
    
    # Save Excel
    wb.save(output_excel)
    print(f"✅ Exported estimate to {output_excel}")

    # --- Generate PDF ---
    page_width, page_height = landscape(A2)
    table_width = page_width * 0.90

    frame = Frame(page_width * 0.05,  # Left margin
                  60,                 # Bottom margin (leave space for footer)
                  table_width,        # Content width
                  page_height - 120,  # Content height (leave space for header & footer)
                  id='main_frame')
    template = PageTemplate(id='custom', frames=[frame], onPage=lambda canvas, doc: header_footer(canvas, doc, project_title, pdf_total_pages))
    
    table = Table(detail_table_data, repeatRows=1)
    
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
    
    summary_table = Table(summary_table_data, repeatRows=1)
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

    table_elements.append(summary_table)
    table_elements.append(PageBreak())
    table_elements.append(table)

    dummy_elements = []
    dummy_elements.append(summary_table)
    dummy_elements.append(PageBreak())
    dummy_elements.append(table)
    dummy_doc = SimpleDocTemplate("dummy.pdf", pagesize=landscape(A2))
    dummy_doc.addPageTemplates([template])
    dummy_doc.build(dummy_elements, onLaterPages=lambda canvas, doc: count_pages(canvas, doc), onFirstPage=lambda canvas, doc: count_pages(canvas, doc))
    print("pdf total pages=======>", pdf_total_pages)
    
    doc = SimpleDocTemplate(output_pdf, pagesize=landscape(A2))
    doc.addPageTemplates([template])
    doc.build(table_elements, onFirstPage=lambda canvas, doc: header_footer(canvas, doc, project_title, pdf_total_pages), onLaterPages=lambda canvas, doc: header_footer(canvas, doc, project_title, pdf_total_pages))

# =================== MAIN PDF PROCESSING WITH LIVE PROGRESS ===================
def get_page_count(pdf_file):
    reader = PdfReader(pdf_file)
    return len(reader.pages)
def start_pdf_processing(pdf_path: str, output_excel, output_pdf, location, currency, session_id, cad_title, unit, prompt):
    combined_text = ""
    txt_path = pdf_path.replace(".pdf", ".txt")
    json_path = pdf_path.replace(".pdf", ".json")
    

    total_pages = get_page_count(pdf_path)

    # if not os.path.exists(txt_path):
    #     print("File does not exist")

        
    all_texts = [""] * total_pages

    def process_page(page_num):
        img_path = convert_pdf_page_to_image(pdf_path, page_num)
        if img_path:
            all_texts[page_num-1] = extract_text_from_image(img_path)
        progress = round((page_num / total_pages) * 100, 2)
        notify_frontend(total_pages, page_num, "Processing is in progress...", "", "", session_id) 

    # 🧵 Run pages concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_page, range(1, total_pages + 1))

    # combined_text = " ".join(all_texts)
    # with open(txt_path, "w", encoding="utf-8", errors="ignore") as file:
    #     file.write(combined_text)

    # 1. Chunk OCR dynamically by tokens
    chunks = chunk_ocr_text_tokenwise(all_texts)
    with open("chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("Chunks length:", len(chunks))
    stage1_items = []
    all_jobs = []
    # 2. extraction per chunk
    for chunk in chunks:
        extracted = get_construction_jobs(chunk['ocr_text'], None, cad_title, location, unit, prompt)
        jobs_list = json.loads(extracted)
        all_jobs.extend(jobs_list)

    with open("result.json", "w") as f:
        json.dump(all_jobs, f, indent=2)

    # ✅ Merge duplicate Job Activities across pages
    merged = {}
    for item in all_jobs: #all_results
        job_name = item.get("Job Activity", "").strip().lower()
        if not job_name:
            continue

        if job_name not in merged:
            merged[job_name] = item
        else:
            for key in [
                "Category", "Quantity"
            ]:
                try:
                    merged[job_name][key] = float(merged[job_name].get(key, 0)) + float(item.get(key, 0))
                except (ValueError, TypeError):
                    pass

    with open("merged.json", "w") as f:
        json.dump(merged, f, indent=2)

    final_jobs_list = list(merged.values())
    final_jobs_list = normalize_categories(final_jobs_list)

    # ✅ Generate Excel output
    if final_jobs_list:
        final_output = generate_summary_from_details(final_jobs_list)
        generate_outputs(final_output, cad_title, currency, output_excel, output_pdf)

        base_path = r"/var/Django/cadProcessor/media"
        pdf_url=os.path.relpath(output_pdf, base_path).replace("\\", "/")
        excel_url = os.path.relpath(output_excel, base_path).replace("\\", "/")

        notify_frontend(total_pages, total_pages, "Processing was completed", pdf_url, excel_url, session_id)
        print("✅ PDF processing completed successfully.")
    else:
        print("⚠️ No valid results extracted from PDF.")

def normalize_categories(data: list) -> list:
    """
    Validate categories, assign division order (Div), 
    and sort by Div following the allowed category order.
    """

    # Map category name to its division number
    # category_to_div = {cat: i + 1 for i, cat in enumerate(allowed_categories)}

    normalized = []
    for item in data:
        cat = str(item.get("Category", "")).strip()
        if cat not in allowed_categories:
            continue

        normalized_item = {**item}
        # normalized_item["Category"] = cat
        normalized_item["Div"] = item.get("CSI Code", "")[:2].strip()  # First two digits of CSI Code as Div

        normalized.append(normalized_item)

    # Sort by Div number
    normalized.sort(key=lambda x: x["Div"])

    return normalized

def generate_summary_from_details(details: list) -> dict:
    """
    Generate a cost summary grouped by Category from the Details list.
    Returns a dict: { "Summary": [...], "Details": details }
    """
    if not details:
        return {"Summary": [], "Details": []}

    summary_map = {}
    total_project_cost = 0.0

    for item in details:
        category = item.get("Category", "Uncategorized").strip()
        div = item.get("Div", 0)
        csi = item.get("CSI Code", "")
        ucost = item.get("M.UCost", 0)
        qty = item.get("Quantity", 0)
        lrate = item.get("L.Rate", 0)
        lhrs = item.get("L.Hrs", 0)
        erate = item.get("E.Rate", 0)
        ehrs = item.get("E.Hrs", 0)
        mcost = round(ucost * qty, 2)
        lcost = round(lrate * lhrs, 2)
        ecost = round(erate * ehrs, 2)
        sub_markups = round((mcost + lcost + ecost) * 0.25, 2)
        subtotal = round(mcost + lcost + ecost + sub_markups, 2)
        key = (div, category)
        summary_map[key] = summary_map.get(key, 0.0) + subtotal
        total_project_cost += subtotal

    # Build summary list
    summary_list = [
        {
            "Div": div,
            "Category": category,
            "Total Cost": round(total, 2)
        }
        for (div, category), total in summary_map.items()
    ]

    # Sort by Div order
    summary_list.sort(key=lambda x: x["Div"])

    # Add total project summary (optional Div for consistency)
    summary_list.append({
        "Div": "",  # Keeps it last
        "Category": "Total Project Cost",
        "Total Cost": round(total_project_cost, 2)
    })

    return {
        "Summary": summary_list,
        "Details": details
    }

def preprocess_cad_text(cad_text: str) -> str:
    """
    Preprocess CAD text to preserve multi-line item relationships while cleaning for AI processing.
    - Preserves line breaks for multi-line items
    - Groups related lines together
    - Cleans excessive whitespace while maintaining structure
    """

    text = cad_text.encode('utf-8', 'ignore').decode()
    text = text.encode('ascii', 'ignore').decode()
    # text = re.sub(r'\\[uU]\w{4}', ' ', text)
    text = re.sub(r"[\u2013\u2014\u2018\u2019\u201c\u201d]", "", text)
    
    # # Remove unwanted symbols and multiple spaces
    text = re.sub(r"[\[\]\{\}\|]", "", text)
    # text = re.sub(r'\s+', '', text)
    
    return text.strip()

def extract_json_from_response(response_text: str):
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match: return match.group(1)
    match2 = re.search(r'\[.*\]', response_text, re.DOTALL)
    return match2.group(0) if match2 else None

def validate_data(data: list) -> bool:
    return isinstance(data, list) and len(data) > 0

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
        if current_page == 0:
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
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ---------------- TOKEN COUNT ----------------
def count_tokens(text: str, model: str = MODEL) -> int:
    """
    Counts tokens using tiktoken.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ---------------- DYNAMIC TOKEN CHUNKING ----------------
def chunk_ocr_text_tokenwise(ocr_pages: List[str], max_tokens: int = MAX_TOKENS_PER_CHUNK) -> List[Dict]:
    """
    Splits OCR pages into chunks by token count.
    Returns list of dicts with metadata.
    """
    chunks = []
    current_chunk_pages = []
    current_tokens = 0
    chunk_id = 1

    for i, page_text in enumerate(ocr_pages):
        page_tokens = count_tokens(page_text)

        # Check if adding this page exceeds max_tokens
        if current_tokens + page_tokens > max_tokens and current_chunk_pages:
            # Save current chunk
            chunk_text = "\n".join(current_chunk_pages)
            chunks.append({
                "chunk_id": f"CHUNK_{chunk_id:03d}",
                "page_range": f"{i - len(current_chunk_pages) + 1}-{i}",
                "ocr_text": chunk_text
            })
            chunk_id += 1
            current_chunk_pages = []
            current_tokens = 0

        # Add page to current chunk
        current_chunk_pages.append(page_text)
        current_tokens += page_tokens

    # Add last chunk
    if current_chunk_pages:
        chunk_text = "\n".join(current_chunk_pages)
        chunks.append({
            "chunk_id": f"CHUNK_{chunk_id:03d}",
            "page_range": f"{len(ocr_pages) - len(current_chunk_pages) + 1}-{len(ocr_pages)}",
            "ocr_text": chunk_text
        })

    return chunks
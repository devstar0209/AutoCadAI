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
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# =================== CONFIG ===================
API_KEY = ""
client = openai.OpenAI(api_key=API_KEY)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Linux
# pytesseract.pytesseract.tesseract_cmd = r"C:\Path\To\tesseract.exe"  # Windows
MAX_WORKERS = 4
_currency = "USD"

allowed_categories = [
    "General Requirements",
    "Existing Conditions",
    "Concrete",
    "Masonry",
    "Metal",
    "Wood, Plastic & Composites",
    "Thermal & Moisture Protection",
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

category_keywords = {

    "Existing Conditions": [
        "remain","demolish", "protect", "survey", 
        "remove", "salvage"
    ],

    "Concrete": [
        "concrete", "paving", "slab", "footing", "foundation", "patch", "curb", 
        "sidewalk", "masonry grout", "reinforcing", "rebar", "cast-in-place"
    ],

    "Masonry": [
        "masonry", "brick", "CMU", "stone", "veneers", "grout", 
        "mortar", "joint", "wall unit", "prefabricated masonry"
    ],

    "Metal": [
        "steel", "metal", "structural", "beam", "column", "angle", "channel", 
        "dowel", "weld", "bracket", "stainless steel", "guardrail", "handrail"
    ],

    "Wood, Plastic & Composites": [
        "wood", "plywood", "composite", "plastic", "millwork", "laminate", 
        "finish carpentry", "paneling", "cabinet", "trim"
    ],

    "Thermal & Moisture Protection": [
        "insulation", "vapor barrier", "waterproof", "membrane", "roofing", 
        "sealant", "caulk", "weatherproof", "thermal protection", "moisture barrier"
    ],

    "Openings": [
        "door", "window", "frame", "hardware", "curtain wall", "glazing", 
        "hatch", "shutter", "access panel", "louvers"
    ],

    "Finishes": [
        "paint", "coating", "tile", "carpet", "flooring", "plaster", "wall covering", 
        "ceiling", "stain", "veneer", "finish", "resilient flooring", "epoxy"
    ],

    "Specialties": [
        "signage", "toilet accessory", "lockers", "whiteboard", "fire extinguisher", "paper holder",
        "flagpole", "bicycle rack", "specialty item", "marker board", "wall mirror", "casework", "medicine chest",
        "closed shelving"
    ],

    "Equipment": [
        "turnstile", "entry", "gate", "fountain", "bench", "exercise equipment", "kitchen equipment",
        "generator", "HVAC unit", "elevator equipment"
    ],

    "Furnishing": [
        "furniture", "desk", "chair", "table", "cabinet", "shelving", "fixture", 
        "casework", "window treatment", "curtain"
    ],

    "Special Construction": [
        "special structure", "swimming pool", "roof top platform", "platform", 
        "greenhouse", "sound barrier", "temporary structure"
    ],

    "Conveying Systems": [
        "elevator", "escalator", "conveyor", "lift", "dumbwaiter", "moving walkway", 
        "hoist", "vertical transport"
    ],

    "Fire Suppression": [
        "sprinkler", "fire pump", "standpipe", "fire line", "hydrant", 
        "fire alarm system", "fire protection", "extinguisher", "suppression system"
    ],

    "Plumbing": [
        "pipe", "valve", "plumbing fixture", "drain", "water supply", "sanitary", 
        "storm drain", "trap", "plumbing line", "hot water", "cold water"
    ],

    "HVAC": [
        "duct", "air handler", "vent", "diffuser", "chiller", "heating", 
        "cooling", "fan", "AHU", "thermostat", "grille", "HVAC equipment"
    ],

    "Electrical": [
        "panelboard", "circuit", "breaker", "wire", "receptacle", 
        "lighting", "switch", "transformer", "distribution", "power", "ATS"
    ],

    "Communications": [
        "conduit", "cable", "jack", "network", "fiber", "telecom", 
        "SYSTIMAX", "riser", "outlet", "structured cabling"
    ],

    "Electronic Safety & Security": [
        "camera", "CCTV", "security system", "access control", "alarm", "sensor", 
        "turnstile", "gate", "motion detector", "security panel"
    ],

    "Earthwork": [
        "excavation", "grading", "cut", "fill", "soil", "compaction", "trenching", 
        "backfill", "site prep", "earth", "subgrade"
    ],

    "Exterior Improvement": [
        "landscape", "sidewalk", "curb", "fence", "scrim", 
        "site furniture", "bollard", "planter", "hardscape"
    ],

    "Utilities": [
        "water line", "sewer line", "gas line", "stormwater", "utility trench", 
        "manhole", "utility connection"
    ],

    "Transportations": [
        "roadway", "pavement", "highway", "parking lot", "striping", 
        "traffic sign", "guardrail", "curb ramp", "transportation"
    ],

    "Waterway & Marine": [
        "dock", "pier", "bulkhead", "marina", "jetty", "boat ramp", "waterway", 
        "wharf", "pile", "marine structure"
    ],

    "Material Processing & Handling Equipment": [
        "conveyor", "crusher", "hopper", "mixing equipment", "processing", 
        "handling system", "material handling", "silo", "bin", "industrial equipment"
    ],

    "Pollution Control Equipment": [
        "scrubber", "emission control", "dust collector", 
        "wastewater treatment", "pollution control", "environmental equipment"
    ]
}


# =================== FRONTEND NOTIFY ===================
def notify_frontend(event_type, **kwargs):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "pdf_processing",
        {"type": event_type, **kwargs}
    )

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
            cleaned_lines.append(line)

        structured_text = "\n".join(cleaned_lines)
        return structured_text.strip()

    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return ""

# =================== AI COST ESTIMATION ===================


def should_use_nrm2(project_location=None, cad_text=""):
    """Determine if NRM2 standards should be used based on project characteristics"""

    # Caribbean countries that commonly use NRM2/RICS standards
    caribbean_countries = [
        "barbados", "trinidad", "tobago", "jamaica", "bahamas", "grenada",
        "st lucia", "dominica", "antigua", "barbuda", "st kitts", "nevis",
        "st vincent", "grenadines", "belize", "guyana", "suriname"
    ]

    # Commonwealth countries that may use NRM2
    commonwealth_indicators = [
        "commonwealth", "rics", "nrm2", "british", "uk standard",
        "metres", "cubic metres", "square metres"
    ]

    # Check project location
    if project_location:
        location_lower = project_location.lower()
        if any(country in location_lower for country in caribbean_countries):
            return True

    # Check CAD text for indicators
    # cad_lower = cad_text.lower()

    # # Look for metric units (strong indicator of NRM2)
    # metric_indicators = ["m³", "m²", "metres", "cubic metres", "square metres", "linear metres"]
    # if any(indicator in cad_lower for indicator in metric_indicators):
    #     return True

    # # Look for NRM2/RICS references
    # if any(indicator in cad_lower for indicator in commonwealth_indicators):
    #     return True

    # # Look for Caribbean location references in CAD text
    # if any(country in cad_lower for country in caribbean_countries):
    #     return True

    return False


def extract_project_location(cad_text):
    """Extract project location from CAD text to determine if NRM2 should be used"""
    if not cad_text:
        return 'USA'

    text_lower = cad_text.lower()

    # Caribbean countries and territories
    caribbean_locations = {
        "barbados": "Barbados",
        "trinidad": "Trinidad and Tobago",
        "tobago": "Trinidad and Tobago",
        "jamaica": "Jamaica",
        "bahamas": "Bahamas",
        "grenada": "Grenada",
        "st lucia": "Saint Lucia",
        "saint lucia": "Saint Lucia",
        "dominica": "Dominica",
        "antigua": "Antigua and Barbuda",
        "barbuda": "Antigua and Barbuda",
        "st kitts": "Saint Kitts and Nevis",
        "saint kitts": "Saint Kitts and Nevis",
        "nevis": "Saint Kitts and Nevis",
        "st vincent": "Saint Vincent and the Grenadines",
        "saint vincent": "Saint Vincent and the Grenadines",
        "grenadines": "Saint Vincent and the Grenadines",
        "belize": "Belize",
        "guyana": "Guyana",
        "suriname": "Suriname"
    }

    # Check for location indicators
    for location_key, location_name in caribbean_locations.items():
        if location_key in text_lower:
            return location_name

    # Check for other Commonwealth indicators
    commonwealth_indicators = ["commonwealth", "rics", "nrm2", "british standard"]
    for indicator in commonwealth_indicators:
        if indicator in text_lower:
            return "Commonwealth"

    return 'USA'


def get_construction_jobs(cad_text, category, project_location='USA'):
    print(f"Starting construction category {category} jobs analysis...")

    # Determine if NRM2 standards should be used
    use_nrm2 = should_use_nrm2(project_location, cad_text)
    # print(f"Using NRM2 standards: {use_nrm2}")

    # Build system prompt based on standards to use

    system_prompt = """
You are an expert construction cost estimator.
Your job is to extract job activities only from the OCR text of CAD drawings and provide detailed cost estimates.
Only use the information explicitly present in the OCR text provided.
"""


    user_prompt = f"""
Extract all job activities from the following OCR text for category: {category}.
For each activity:
- Skip exact duplicate activities
- Use 2024 MasterFormat (CSI) codes
- Assign one of the allowed categories from the predefined list: {','.join(allowed_categories)}
- Adjust units according to : 
   • Caribbean/Commonwealth → NRM2 units
   • Otherwise → RSMeans US units. For RSMeans US: Concrete Paving → SF; other concrete → CY.

Return a JSON array of objects, one per activity, with these fields:
- CSI Code (format 01 02 03.04)
- Category
- Job Activity (only exact or relative description as it appears in OCR text)
- Quantity (number, use default only if missing)
- Unit (use default only if missing)
- M.UCost (material cost per unit)
- L.Rate (labor rate per hour)
- L.Hrs (round number)
- E.Rate (equipment rate per hour)
- E.Hrs (round number)
- Currency

Project Location: {project_location}

OCR text:
{cad_text}
"""

    try:
        response = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:global-precisional-services-llc::CSwogr3u",
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
        print(f"AI response received:: {response_text}")
        
        return response_text
    except Exception as e:
        print(f"Error in get_construction_jobs: {e}")
        return ""

# =================== PDF PAGE PROCESSING ===================
def convert_pdf_page_to_image(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(pdf_path, dpi=200, first_page=page_number, last_page=page_number, poppler_path="/usr/bin")
    if not images: return ""
    directory = os.path.dirname(pdf_path)
    image_path = os.path.join(directory, f"page_{page_number}.png")
    images[0].save(image_path, "PNG")
    return image_path

# =================== OUTPUT GENERATION ===================
def generate_outputs(output_json: dict, filename: str):
    print(f"Generating outputs...")
    """
    Save structured JSON estimate into Excel with Summary + Details.
    """
    wb = Workbook()

    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    ws_summary = wb.active
    ws_summary.title = "Summary"

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

    ws_summary.append(["Div", "Description", "Total"]) # headers

    # Style header
    for col in range(1, 4):
        cell = ws_summary.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border_style
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Set column widths
    ws_summary.column_dimensions['A'].width = 5
    ws_summary.column_dimensions['B'].width = 50
    ws_summary.column_dimensions['C'].width = 18

    for item in output_json.get("Summary", []):
        currency = item.get("Currency", "USD")
        ws_summary.append([
        item.get("Div", 0),
        item.get("Category", ""),
        currency+"$"+str(format(item.get("Total Cost", 0), ","))
    ])

    # --- Write Details Sheet ---
    ws_details = wb.create_sheet("Details")
    headers = ["Div", "CSI Code", "Description", "Quant.", "Unit", "M.U/Cost", "Total.Material", "Equip.Hrs","Equip.Rate", "Total.Equip","Labor.Hrs","Labor.Rate", "Total.Labor", "Sub Markups", "Subtotal Cost"]
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
        currency = item.get("Currency", "USD")
        ucost = round(item.get("M.UCost", 0), 2)
        qty = item.get("Quantity", 0)
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
            item.get("Unit", ""),
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
    wb.save(filename)
    print(f"✅ Exported estimate to {filename}")

# =================== MAIN PDF PROCESSING WITH LIVE PROGRESS ===================
def get_page_count(pdf_file):
    reader = PdfReader(pdf_file)
    return len(reader.pages)
def start_pdf_processing(pdf_path: str, output_excel, output_pdf, location, currency, session_id, cad_title):
    combined_text = ""
    txt_path = pdf_path.replace(".pdf", ".txt")
    json_path = pdf_path.replace(".pdf", ".json")
    _currency = currency

    if not os.path.exists(txt_path):
        print("File does not exist")

        total_pages = get_page_count(pdf_path)
        
        all_texts = [""] * total_pages

        def process_page(page_num):
            img_path = convert_pdf_page_to_image(pdf_path, page_num)
            if img_path:
                all_texts[page_num-1] = extract_text_from_image(img_path)
            progress = round((page_num / total_pages) * 100, 2)
            notify_frontend(
                "page_processed",
                page=page_num,
                total_pages=total_pages,
                progress=progress
            )

        # 🧵 Run pages concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(process_page, range(1, total_pages + 1))

        combined_text = " ".join(all_texts)
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(combined_text)

    category_text = {cat: [] for cat in category_keywords}

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line_clean = line.strip().lower()
            if not line_clean:
                continue
            if line_clean in {"gate"}:
                continue
            line_clean = preprocess_cad_text(line_clean)
            for cat, keywords in category_keywords.items():
                for k in keywords:

                    # Escape keyword, enforce word boundaries
                    pattern = r"\b" + re.escape(k.lower()) + r"\b"

                    if re.search(pattern, line_clean):
                        category_text[cat].append(line_clean)
                        break
                    else:
                        continue  # inner loop not matched
                    break  # outer loop matched

    with open(json_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(category_text))

    combined_category_text = {
        cat: " ".join(lines)
        for cat, lines in category_text.items()
        if lines
    }

    all_jobs = []
    for cat, text in combined_category_text.items():
        if not text.strip():
            continue

        if location is None:
            # Try to extract project location from text
            location = extract_project_location(text)
        
        try:
            # Call AI function
            res = get_construction_jobs(text, cat, location)
            jobs_list = json.loads(res)  # Try parsing JSON
            if isinstance(jobs_list, list):
                all_jobs.extend(jobs_list)  # Add to combined list
            else:
                print(f"Warning: JSON is not a list for category {cat}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for category {cat}: {e}")
            print("Response:", res)
            continue  # skip this category if JSON invalid

        print(f"Total jobs collected: {len(all_jobs)}")

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

    final_jobs_list = list(merged.values())
    final_jobs_list = normalize_categories(final_jobs_list)

    # ✅ Generate Excel output
    if final_jobs_list:
        final_output = generate_summary_from_details(final_jobs_list)
        generate_outputs(final_output, output_excel)
        notify_frontend(
            "pdf_processing_completed",
            pdf_path=output_pdf,
            excel_path=output_excel,
            progress=100
        )
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
        normalized_item["Category"] = cat
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
    currency = "USD"

    for item in details:
        category = item.get("Category", "Uncategorized").strip()
        div = item.get("Div", 0)
        currency = item.get("Currency", "USD")
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
        key = (div, category, currency)
        summary_map[key] = summary_map.get(key, 0.0) + subtotal
        total_project_cost += subtotal

    # Build summary list
    summary_list = [
        {
            "Div": div,
            "Category": category,
            "Currency": currency,
            "Total Cost": round(total, 2)
        }
        for (div, category, currency), total in summary_map.items()
    ]

    # Sort by Div order
    summary_list.sort(key=lambda x: x["Div"])

    # Add total project summary (optional Div for consistency)
    summary_list.append({
        "Div": "",  # Keeps it last
        "Category": "Total Project Cost",
        "Currency": currency,
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
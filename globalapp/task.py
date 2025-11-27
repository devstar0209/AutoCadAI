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
API_KEY = "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"
client = openai.OpenAI(api_key=API_KEY)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Linux
# pytesseract.pytesseract.tesseract_cmd = r"C:\Path\To\tesseract.exe"  # Windows
MAX_WORKERS = 4

# CSI division-based keywords
construction_keywords = {
    "03 - Concrete": ["concrete", "slab", "footing", "foundation", "column", "beam", "girder",
                      "pile", "pier", "rebar", "reinforcement", "formwork", "joint", "curing"],
    "04 - Masonry": ["masonry", "brick", "block", "cmu", "stone", "veneer", "grout", "mortar", "lintel"],
    "05 - Metals": ["steel", "weld", "bolt", "plate", "angle", "channel", "pipe", "tube", "joist", "deck"],
    "06 - Wood": ["wood", "lumber", "timber", "plywood", "osb", "truss", "joist", "stud", "sheathing"],
    "07 - Thermal & Moisture": ["roof", "roofing", "membrane", "insulation", "vapor barrier",
                                "sealant", "flashing", "shingle", "tile"],
    "08 - Openings": ["door", "window", "frame", "glazing", "curtain wall", "skylight"],
    "09 - Finishes": ["floor", "ceiling", "tile", "carpet", "paint", "coating", "plaster",
                      "gypsum", "drywall", "veneer", "paneling"],
    "21 - Fire Protection": ["sprinkler", "fire protection", "standpipe", "fire pump", "alarm"],
    "22 - Plumbing": ["plumbing", "pipe", "valve", "toilet", "sink", "water heater", "drainage", "fixture"],
    "23 - HVAC": ["hvac", "duct", "chiller", "boiler", "air handler", "diffuser", "damper", "ventilation"],
    "26 - Electrical": ["electrical", "conduit", "cable", "wire", "panel", "transformer", "lighting",
                        "outlet", "switch", "breaker", "generator", "feeder", "grounding", "data", "telecom"],
    "Measurement Units": ["dimension", "length", "width", "height", "depth", "elevation", "level", "slope",
                          "thickness", "diameter", "radius", "area", "volume", "square", "cubic", "linear",
                          "feet", "foot", "inch", "inches", "meter", "millimeter", "centimeter"]
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
        return None

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

    return None


def get_construction_jobs(cad_text, project_location=None):
    print(f"Starting construction jobs analysis...")

    cad_text = preprocess_cad_text(cad_text)

    print(f"preprocess_cad_text: {cad_text}")

    # Determine if NRM2 standards should be used
    use_nrm2 = should_use_nrm2(project_location, cad_text)
    print(f"Using NRM2 standards: {use_nrm2}")

    # Build system prompt based on standards to use

    system_prompt = f"""
    You are an expert construction estimator. 
Your job is to read unstructured OCR text from drawings and extract each job activity into structured JSON. 
Each activity must include: Category, Job Activity, Quantity, Unit, Rate, Material Cost, Labor Cost, Equipment Cost, Sub Markups, Subtotal Cost.
When OCR text lacks explicit quantities or rates, assume reasonable defaults so that every activity still produces a complete JSON record. Do not refuse due to missing data.
You are only allowed to use one of the following categories exactly as written: "General Requirements", "Existing Conditions", "Concrete", "Masonry", "Metal", "Wood, Plastic & Composites", "Thermal & Moisture Protection", "Openings", "Finishes", "Specialties", "Equipment", "Furnishing", "Special Construction", "Conveying Systems", "Fire Suppression", "Plumbing", "HVAC", "Electrical", "Communications", "Electronic Safety & Security", "Earthwork", "Exterior Improvement", "Utilities", "Transportations", "Waterway & marine", "Material Processing & Handling Equipment", "Pollution Control Equipment"
Do NOT create new or modified categories. If the text does not match any category, skip.
    """

    user_prompt = f"""Extract all job activities and costs from the following text.
Return JSON array of objects, one per activity. 
- For each activity, assign one of the allowed categories from the predefined list. If no category applies, skip.
- SubTotal Cost = Material Cost + Labor Cost + Equipment Cost + Sub Markups. Round numbers
- Default Cost Allocation if breakdown unknown:
    • Material: 50–65%
    • Labor: 30–40%
    • Equipment: 5–15%
    • Sub Markups: 12–20%
- OCR text:
{cad_text}

"""


    try:
        response = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:global-precisional-services-llc::CSwogr3u",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=0.1,
            store= True,
        )
        response_text = response.choices[0].message.content

        # response = client.responses.create(
        #     model="ft:gpt-4o-2024-08-06:global-precisional-services-llc::CRoIewoT",
        #     input=[    
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     temperature=0.2
        # )
        # response_text = response.output_text

        # Remove markdown fences and extra whitespace
        response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.DOTALL).strip()
        print(f"AI response received:: {response_text}")
        # response_text = response_text.replace("\n", "").replace("\r", "")

        # Remove trailing commas if needed
        # response_text = re.sub(r",(\s*[}\]])", r"\1", response_text)
        
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
    # if "Summary" in wb.sheetnames:
    #     ws_summary = wb["Summary"]
    # else:
    #     ws_summary = wb.create_sheet("Summary")

    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_summary.append(["Div", "Description", "Cost"]) # headers

    for item in output_json.get("Summary", []):
        ws_summary.append([
        item.get("Div", 0),
        item.get("Category", ""),
        item.get("Total Cost", 0)
    ])
    # ws_summary.append(["Total", output_json["Summary"].get("Total", 0)])

    # --- Write Details Sheet ---
    ws_details = wb.create_sheet("Details")
    headers = ["Div", "Description", "Quantity", "Unit", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Sub Markups", "Subtotal Cost"]
    ws_details.append(headers)

    for item in output_json.get("Details", []):
        ws_details.append([
            item.get("Div", 0),
            item.get("Job Activity", ""),
            item.get("Quantity", ""),
            item.get("Unit", ""),
            item.get("Rate", ""),
            item.get("Material Cost", ""),
            item.get("Equipment Cost", ""),
            item.get("Labor Cost", ""),
            item.get("Sub Markups", ""),
            item.get("Subtotal Cost", "")
        ])

    # Save Excel
    wb.save(filename)
    print(f"✅ Exported estimate to {filename}")

# =================== MAIN PDF PROCESSING WITH LIVE PROGRESS ===================
def get_page_count(pdf_file):
    reader = PdfReader(pdf_file)
    return len(reader.pages)
def start_pdf_processing(pdf_path: str, output_pdf: str, output_excel: str, location=None):
    total_pages = get_page_count(pdf_path)
    # all_results = []  # store combined structured outputs

    # def process_page(page_num):
    #     img_path = convert_pdf_page_to_image(pdf_path, page_num)
    #     if not img_path:
    #         return ""

    #     # OCR for this page
    #     page_text = extract_text_from_image(img_path)
    #     if not page_text.strip():
    #         return ""

    #     print(f"🔹 Running AI extraction on page {page_num}/{total_pages}...")
    #     try:
    #         # Pass per-page text to AI (your existing fine-tuned function)
    #         page_result = get_construction_jobs(page_text, location)

    #         # Parse and validate JSON
    #         if isinstance(page_result, str):
    #             try:
    #                 page_result = json.loads(page_result)
    #             except json.JSONDecodeError:
    #                 print(f"⚠️ Invalid JSON on page {page_num}, skipping.")
    #                 return ""

    #         # Append valid structured results
    #         if isinstance(page_result, list):
    #             all_results.extend(page_result)

    #     except Exception as e:
    #         print(f"⚠️ Error processing page {page_num}: {e}")

    #     # Notify frontend
    #     progress = round((page_num / total_pages) * 100, 2)
    #     notify_frontend(
    #         "page_processed",
    #         page=page_num,
    #         total_pages=total_pages,
    #         progress=progress
    #     )
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
    # Try to extract project location from PDF metadata or text
    # project_location = extract_project_location(combined_text)
    res = get_construction_jobs(combined_text, location)
    jobs_list = json.loads(res)

    # ✅ Merge duplicate Job Activities across pages
    merged = {}
    for item in jobs_list: #all_results
        job_name = item.get("Job Activity", "").strip().lower()
        if not job_name:
            continue

        if job_name not in merged:
            merged[job_name] = item
        else:
            for key in [
                "Category", "Quantity", "Rate", "Material Cost", "Labor Cost",
                "Equipment Cost", "Sub Markups", "Subtotal Cost"
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

    # Map category name to its division number
    category_to_div = {cat: i + 1 for i, cat in enumerate(allowed_categories)}

    normalized = []
    for item in data:
        cat = str(item.get("Category", "")).strip()
        if cat not in allowed_categories:
            continue

        normalized_item = {**item}
        normalized_item["Category"] = cat
        normalized_item["Div"] = category_to_div[cat]

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

        try:
            subtotal = float(item.get("Subtotal Cost", 0))
        except (TypeError, ValueError):
            subtotal = 0.0

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
        "Div": 99,  # Keeps it last
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
    text = re.sub(r'\\[uU]\w{4}', ' ', text)
    
    # Remove unwanted symbols and multiple spaces
    text = re.sub(r'[^ -~\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_json_from_response(response_text: str):
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match: return match.group(1)
    match2 = re.search(r'\[.*\]', response_text, re.DOTALL)
    return match2.group(0) if match2 else None

def validate_data(data: list) -> bool:
    return isinstance(data, list) and len(data) > 0

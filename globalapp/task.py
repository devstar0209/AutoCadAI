import os
import re
import json
import cv2
import pytesseract
import openai
import openpyxl
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from collections import defaultdict, Counter

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
    try:
        if not os.path.exists(image_path):
            return ""

        img = cv2.imread(image_path)
        if img is None:
            return ""
        print(f"Processing OCR function...: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        denoised = cv2.medianBlur(enhanced, 3)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        processed_images = [gray, denoised, thresh]
        all_texts = []

        for proc_img in processed_images:
            text = pytesseract.image_to_string(proc_img, config="--psm 6")
            # rotated = cv2.rotate(proc_img, cv2.ROTATE_90_CLOCKWISE)
            # rotated_text = pytesseract.image_to_string(rotated, config='--psm 6')
            # text = origin_text + " " + rotated_text
            if text and len(text.strip()) > 0:
                all_texts.append(text)
        # Select the best result
        best_text = ""
        max_score = 0

        for text in all_texts:
            score = len(text.strip())  # base score: text length
            text_lower = text.lower()
            # Add score for keyword matches
            for division, keywords in construction_keywords.items():
                for kw in keywords:
                    if re.search(rf"\b{re.escape(kw.lower())}\b", text_lower):
                        score += 10
            if score > max_score:
                max_score = score
                best_text = text

        if best_text:
            cleaned_text = re.sub(r"\s+", " ", best_text.strip())
            return cleaned_text

        return ""

    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return ""

# =================== AI COST ESTIMATION ===================

def structure_cad_text_with_ai(cad_text: str):
    """
    Use AI to convert raw OCR text into a normalized, machine-readable structure
    grouped by job category. Returns a Python object (list/dict) or None on failure.

    Expected JSON (examples):
    {
      "Concrete": [{"item":"SLAB","quantity":150,"unit":"CY"}, {"item":"FOOTING","quantity":80,"unit":"CY"}],
      "Masonry": [{"item":"CMU WALL","quantity":1200,"unit":"SF"}],
      ...
    }
    """
    try:
        struct_prompt = (
            "You are a construction plans takeoff assistant. Read the CAD OCR text and return ONLY valid JSON grouped by job category.\n"
            "General Rules:\n"
            "- Categories: Concrete, Masonry, Metals, Finishes, Thermal & Moisture Protection, HVAC, Plumbing, Electrical, Fire Protection, Electrical Safety & Security, Sitework, Equipment.\n"
            "- Each category is a list of objects: {\"item\": <string>}\n"
            "- No prose, no markdown, return valid JSON only.\n\n"
            "- Be exhaustive; preserve exact names; separate entries per name/size/rating"
            
            "CATEGORY RULES:\n"
            "- MASONRY: Include CMU walls, BRICK walls, STONE, BLOCK. Use SF (wall area) or CY if thickness is clear.\n"
            "- METALS: Include STRUCTURAL STEEL, BEAMS, COLUMNS, STAIRS, HANDRAILS. Units = TONS (weight) or LF/SF.\n"
            "- FINISHES: Include PAINT (SF), FLOORING (SF), CEILING (SF), DOORS/WINDOWS (EA), MILLWORK.\n"
            "- THERMAL & MOISTURE PROTECTION: Include ROOFING (SF), INSULATION (SF), WATERPROOFING (SF), VAPOR BARRIER.\n"
            "- HVAC: Include DUCTS (LF/SF with size if present), FANS, AHUs, FCUs, AC UNITS (tons), DIFFUSERS, GRILLES.\n"
            "- PLUMBING: Include PIPES (LF, with size/type if present), DRAINS, VENTS, FIXTURES (SINK, TOILET, WATER CLOSET, PUMP).\n"
            "- ELECTRICAL:\n"
            "  • Panels and switchboards: PANEL <name>, DISTRIBUTION SWITCHBOARD <name>, DISTRIBUTION PANEL <name>, SWITCHBOARD <name> (capture AMP if present: 60A–4000A).\n"
            "  • Breakers/MCBs if explicit (by frame or rating).\n"
            "  • Receptacles: DUPLEX, QUAD, GFCI, WEATHERPROOF, FLOOR BOX, etc.\n"
            "  • Switching: SWITCH, 3-WAY, 4-WAY, DIMMER.\n"
            "  • Lighting: LIGHT FIXTURE, TROFFER, DOWNLIGHT, HIGH-BAY, EXIT/EMERG LIGHT, STRIP, WALL PACK, POLE LIGHT.\n"
            "  • Transformers: capture KVA if present.\n"
            "  • Generators: capture GENERATOR.\n"
            "  • Cabling: BRANCH CABLE (capture size/type if present: #12, #10, THHN, XHHW), FEEDER CABLE (capture conductor count and size: 3#350MCM, 4#4/0, 3C-500kcmil).\n"
            "  • Conduit: EMT, RGS, PVC (capture size/type if present)\n"
            "- CAPTURE PATTERNS (ratings/sizes):\n"
            "  • AMP rating: (\"\\b(\d{2,4})\\s*A(MP)?\\b\").\n"
            "  • kVA rating: (\"\\b(\d{1,4}(?:\\.\\d+)?)\\s*KVA\\b\").\n"
            "  • Cable sizes: (#\d{1,2}|\d{1,4}kcmil|MCM|kcmil), with conductor count like 3#350MCM, 4C-500kcmil.\n"
            "  • Conduit sizes: (C-4#1, 2 1/2\"C-A#4/0, 2°C-4#4/0 ...).\n"
            "- CONCRETE: FOUNDATIONS, SLABS (CY), FOOTINGS (CY/LF), PAVING, CONCRETE PAVING, SIDEWALK, DRIVEWAY (slab-on-grade with 6\" default thickness if unspecified; CY or SF with conversion).\n"
            "- HVAC: DUCT (LF/SF with size if present), FAN, AHU/FCU, AC units (tonnage if present), GRILLE/DIFFUSER.\n"
            "- FIRE PROTECTION: Include SPRINKLERS, FIRE ALARMS, HYDRANTS, HOSE CABINETS, SMOKE DETECTORS.\n"
            "- ELECTRICAL SAFETY & SECURITY: Include CAMERAS, ACCESS CONTROL, ALARMS, CARD READERS, INTERCOMS.\n"
            "- SITEWORK: Include EXCAVATION, GRADING, BACKFILL, ASPHALT, SIDEWALKS, CURBS, LANDSCAPING.\n"
            "- EQUIPMENT: Include LEVATORS, LIFTS, etc.\n\n"

            f"CAD_OCR_TEXT:\n{cad_text}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract structured construction items from OCR text as strict JSON by category."},
                {"role": "user", "content": struct_prompt}
            ],
            temperature=0.1,
        )
        text = response.choices[0].message.content
        clean = extract_json_from_response(text) or text
        return json.loads(clean)
    except Exception as e:
        print(f"Structuring AI failed: {e}")
        return None

def get_construction_jobs(cad_text):
    print(f"Starting construction jobs analysis...")
    
    # First, attempt AI structuring of OCR text
    structured_items = structure_cad_text_with_ai(cad_text)

    # Preprocess (existing heuristic summary) as supplemental context
    processed_cad_text = preprocess_cad_text(cad_text)
    
    # Build the cost-estimation prompts
    structured_json_block = json.dumps(structured_items) if structured_items else "{}"
    print(f"structured_json_block text ==> {structured_json_block}")
    
    system_prompt = """You are a professional construction estimator with 20+ years of experience. Analyze CAD text and symbols to produce comprehensive cost estimates.

1. ACCURACY REQUIREMENTS:
   - Use 2024 MasterFormat (CSI) codes
   - Labor ELECTRICAL, Use 2023 - 2024 Edition NECA
   - Base costs on current US market rates (RSMeans-like)
   - Consider regional variation (use national average if unknown)
   - Quantities must be realistic for the described scope and scale
   - Use NRM2 2nd edition UK Oct 2025,  standards method of measurements (if cost estimates is for Caribbean Countries)

2. REQUIRED OUTPUT FORMAT:
   - EXACT JSON list only, no prose: [{"CSI code":"03 30 00","Category":"Concrete","Job Activity":"Cast-in-place Concrete Slab, 6-inch thick","Quantity":150,"Unit":"CY","Rate":125.5,"Material Cost":12000,"Equipment Cost":3000,"Labor Cost":4500,"Total Cost":19500}]
   - All fields are mandatory; values must be numbers for costs/quantities/rate
   - CSI Code MUST be organized in a progressive, six-digit sequence (with further specificity decimal extension, if required).
   - Job Activity MUST be specific and self-contained (type, size, capacity, method). Do NOT embed bracketed notes. Each activity is one line item.

3. COVERAGE AND NAME PRESERVATION (CRITICAL):
   - Use both the STRUCTURED_ITEMS_JSON and the HUMAN_SUMMARIES as ground truth. Do not omit major categories with strong signals.
   - PRESERVE EXACT ITEM NAMES from STRUCTURED_ITEMS_JSON for named equipment/devices.
   - PROPAGATE ATTRIBUTES into the Job Activity text when present in STRUCTURED_ITEMS_JSON:
       • rating (e.g., 125A, 30 kVA, 250 kW)
       • size (e.g., 3#250MCM, 1\" conduit)
   - PRESERVE MULTIPLICITY per distinct name/size/rating: separate rows and correct quantities for each distinct item.

4. DERIVATION RULES (APPLY IF MISSING FROM STRUCTURED_ITEMS_JSON):
   - BRANCH CABLE (LF) = (RECEPTACLE + SWITCH + DIMMER + LIGHT FIXTURE counts) × 12 LF (or 25 LF if long runs are implied). Include as its own line item.
   - FEEDER CABLE (MCM) (LF) = (PANELS + TRANSFORMERS + GENERATORS) × 200 LF. Include as its own line item.
   - CONDUIT (LF) ≈ 9% of total cable length (branch + feeder). Include as its own line item.

5. QUANTITY/RATE/COST RULES (CRITICAL):
   - Total Cost = Material Cost + Labor Cost + Equipment Cost.
   - Rate = Total Cost / Quantity. If an external Rate is provided, treat it as total unit rate and split costs accordingly.
   - Component percentage bands: Material 60–70%, Labor 25–35%, Equipment 5–15%.

6. VALIDATION CHECKLIST:
   - Cover categories with strong signals (Concrete, Masonry, Metals, Finishes, Thermal/Moisture, HVAC, Plumbing, Electrical, Sitework) if present.
   - Positive quantities/costs; math consistent; CSI format "XX XX XX".
   - PRESERVE exact item names, sizes, ratings, and multiplicities for ELECTRICAL equipment/devices/cables/conduit as implied by STRUCTURED_ITEMS_JSON.
"""

    user_prompt = f"""STRUCTURED_ITEMS_JSON:
{structured_json_block}

HUMAN_SUMMARIES:
{processed_cad_text}

REQUIRED ANALYSIS:
1) Expand STRUCTURED_ITEMS_JSON into detailed Job Activities across all detected categories.
2) For each entry in STRUCTURED_ITEMS_JSON, create a corresponding line item that:
   - Uses the exact item name (no generic renaming), includes rating/size when provided.
   - Preserves multiplicities per distinct name/size/rating (separate rows or quantities).
   - Assigns CSI code, realistic quantity/unit, and computes Material/Labor/Equipment/Total costs with Rate.

OUTPUT: ONLY a valid JSON array as specified above, no text.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        response_text = response.choices[0].message.content
        print(f"Initial AI response received:: {response_text}")
        validated_response = validate_and_improve_response(response_text, processed_cad_text)
        return validated_response
    except Exception as e:
        print(f"Error in get_construction_jobs: {e}")
        return get_construction_jobs_fallback(processed_cad_text)

def validate_and_improve_response(response_text, cad_text):
    """Validate AI response and improve if necessary"""
    try:
        # Try to extract and parse JSON
        clean_json = extract_json_from_response(response_text)
        if not clean_json:
            print("No JSON found in response, attempting to extract...")
            # Try to find JSON without code blocks
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
        
        if clean_json:
            parsed_data = json.loads(clean_json)
            
            # Validate the data structure and content
            validation_result = validate_construction_data(parsed_data)
            
            if validation_result["is_valid"]:
                print(f"Response validation passed: {validation_result['score']}/100")
                return parsed_data
            else:
                print(f"Response validation failed: {validation_result['errors']}")
                # Try to improve the response
                return improve_response_with_feedback(response_text, validation_result, cad_text)
        else:
            print("Could not extract valid JSON, using fallback...")
            return get_construction_jobs_fallback(cad_text)
            
    except Exception as e:
        print(f"Validation error: {e}")
        return get_construction_jobs_fallback(cad_text)

def validate_construction_data(data):
    """Validate construction data for accuracy and completeness"""
    errors = []
    score = 100
    
    if not isinstance(data, list):
        errors.append("Data is not a list")
        return {"is_valid": False, "errors": errors, "score": 0}
    
    if len(data) == 0:
        errors.append("No construction activities found")
        return {"is_valid": False, "errors": errors, "score": 0}
    
    required_fields = ["CSI code", "Category", "Job Activity", "Quantity", "Unit", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not a dictionary")
            score -= 20
            continue
            
        # Check required fields
        for field in required_fields:
            if field not in item:
                errors.append(f"Item {i} missing field: {field}")
                score -= 5
        
        # Validate numeric fields
        numeric_fields = ["Quantity", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]
        for field in numeric_fields:
            if field in item:
                try:
                    value = float(item[field])
                    if value <= 0:
                        errors.append(f"Item {i} {field} must be positive")
                        score -= 3
                except (ValueError, TypeError):
                    errors.append(f"Item {i} {field} is not a valid number")
                    score -= 5
        
        # Validate cost relationships
        if all(field in item for field in ["Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]):
            try:
                calculated_total = float(item["Material Cost"]) + float(item["Equipment Cost"]) + float(item["Labor Cost"])
                actual_total = float(item["Total Cost"])
                if abs(calculated_total - actual_total) > 0.01:  # Allow small rounding differences
                    errors.append(f"Item {i} total cost doesn't match sum of components")
                    score -= 10
            except (ValueError, TypeError):
                pass
        
        # Validate CSI code format
        # if "CSI code" in item:
        #     csi_code = str(item["CSI code"])
        #     if not re.match(r'^\d{2}\s\d{2}\s\d{2}$', csi_code):
        #         errors.append(f"Item {i} has invalid CSI code format: {csi_code}")
        #         score -= 5
    
    is_valid = score >= 70 and len(errors) <= 3
    return {"is_valid": is_valid, "errors": errors, "score": score}

def improve_response_with_feedback(original_response, validation_result, cad_text):
    """Improve AI response based on validation feedback"""
    feedback_prompt = f"""The previous response had these issues: {validation_result['errors']}

Please provide a corrected response that addresses these problems. Focus on:
1. Ensuring all required fields are present
2. Making sure all costs are positive numbers
3. Verifying that Total Cost = Material Cost + Equipment Cost + Labor Cost
4. Using proper CSI code format (XX XX XX)
5. Providing realistic quantities and costs

Original CAD text: {cad_text}

Return ONLY a valid JSON array with the exact structure specified in the system prompt."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional construction estimator. Fix the previous response based on the feedback provided."},
                {"role": "user", "content": feedback_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        improved_response = response.choices[0].message.content
        validated_response = validate_and_improve_response(improved_response, cad_text)
        
        print("Improved response generated based on feedback")
        return validated_response
        
    except Exception as e:
        print(f"Error improving response: {e}")
        return get_construction_jobs_fallback(cad_text)

def get_construction_jobs_fallback(cad_text):
    """Fallback method with simpler, more reliable approach"""
    print("Using fallback method for construction jobs...")
    
    fallback_prompt = f"""Based on this CAD drawing text, provide a simple list of common construction activities with basic cost estimates:

{cad_text}

Return a JSON array with this exact format:
[{{"CSI code": "03 30 00", "Category": "Concrete", "Job Activity": "Foundation", "Quantity": 100, "Unit": "CY", "Rate": 120, "Material Cost": 8000, "Equipment Cost": 2000, "Labor Cost": 2000, "Total Cost": 12000}}]

Include only the most obvious construction activities. Keep it simple and accurate."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a construction estimator. Provide simple, accurate cost estimates."},
                    {"role": "user", "content": fallback_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Fallback method also failed: {e}")
        # Return error message instead of default data
        return f'{{"error": "AI analysis failed: {str(e)}", "message": "Unable to process CAD drawing for cost estimation. Please check your OpenAI API key and try again."}}'

# =================== PDF PAGE PROCESSING ===================
def convert_pdf_page_to_image(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(pdf_path, dpi=200, first_page=page_number, last_page=page_number, poppler_path="/usr/bin")
    if not images: return ""
    directory = os.path.dirname(pdf_path)
    image_path = os.path.join(directory, f"page_{page_number}.png")
    images[0].save(image_path, "PNG")
    return image_path

# =================== OUTPUT GENERATION ===================
def generate_outputs(jobs_list: list, output_pdf: str, output_excel: str):
    print(f"Generating outputs...")
    headers = [
        "CSI code","Category","Job Activity","Quantity","Unit","Rate",
        "Material Cost","Equipment Cost","Labor Cost","Total Cost"
    ]
    table_data = [headers]

    material_sum = equipment_sum = labor_sum = total_sum = 0

    for item in jobs_list:
        # Safe parsing
        try:
            quantity = int(item.get("Quantity") or 0)
        except Exception:
            quantity = 0

        row = [
            str(item.get("CSI code","")),
            str(item.get("Category","")),
            str(item.get("Job Activity","")),
            f"{quantity:,}",
            str(item.get("Unit","")),
            f"${float(item.get('Rate') or 0):,.2f}",
            f"${float(item.get('Material Cost') or 0):,.2f}",
            f"${float(item.get('Equipment Cost') or 0):,.2f}",
            f"${float(item.get('Labor Cost') or 0):,.2f}",
            f"${float(item.get('Total Cost') or 0):,.2f}",
        ]
        table_data.append(row)

        material_sum += float(item.get('Material Cost') or 0)
        equipment_sum += float(item.get('Equipment Cost') or 0)
        labor_sum += float(item.get('Labor Cost') or 0)
        total_sum += float(item.get('Total Cost') or 0)

    # Safe summary row (auto matches header length)
    summary_row = [""] * (len(headers) - 5) + [
        "Total",
        f"${material_sum:,.2f}",
        f"${equipment_sum:,.2f}",
        f"${labor_sum:,.2f}",
        f"${total_sum:,.2f}"
    ]
    table_data.append(summary_row)

    # Validate row lengths & fix if needed
    expected_cols = len(headers)
    for i, row in enumerate(table_data):
        if len(row) < expected_cols:
            row.extend([""] * (expected_cols - len(row)))  # pad short rows
        elif len(row) > expected_cols:
            row = row[:expected_cols]  # truncate extras
            table_data[i] = row

    # Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=landscape(A3))
    table = Table(table_data, repeatRows=1)

    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,-1), (-1,-1), colors.lightgrey),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ])
    table.setStyle(style)

    doc.build([table])

    # Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Cost Estimation"
    for row in table_data:
        ws.append(row)
    wb.save(output_excel)

# =================== MAIN PDF PROCESSING WITH LIVE PROGRESS ===================
def get_page_count(pdf_file):
    reader = PdfReader(pdf_file)
    return len(reader.pages)
def start_pdf_processing(pdf_path: str, output_pdf: str, output_excel: str):
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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_page, range(1, total_pages + 1))

    combined_text = " ".join(all_texts)
    jobs_list = get_construction_jobs(combined_text)

    if jobs_list:
        generate_outputs(jobs_list, output_pdf, output_excel)
        notify_frontend(
            "pdf_processing_completed",
            pdf_path=output_pdf,
            excel_path=output_excel,
            progress=100
        )

def preprocess_cad_text(cad_text: str) -> str:
    """
    Generalized text structuring for AI prompts.
    - Extracts quantities per unique item/unit
    - Counts keywords without numbers
    - Categorizes into job categories
    - Builds AI-readable structured text
    """

    text = cad_text.upper()
    text = re.sub(r'\s+', ' ', text)

    return text

def extract_json_from_response(response_text: str):
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match: return match.group(1)
    match2 = re.search(r'\[.*\]', response_text, re.DOTALL)
    return match2.group(0) if match2 else None

def validate_data(data: list) -> bool:
    return isinstance(data, list) and len(data) > 0
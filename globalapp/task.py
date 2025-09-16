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
        text1 = pytesseract.image_to_string(gray, config='--psm 6')
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        denoised = cv2.medianBlur(enhanced, 3)
        text2 = pytesseract.image_to_string(denoised, config='--psm 6')
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        text3 = pytesseract.image_to_string(thresh, config='--psm 6')

        # Combine and clean the results
        all_texts = [text1, text2, text3]
        
        # Select the best result based on length and content quality
        best_text = ""
        max_score = 0
        
        for text in all_texts:
            if text and len(text.strip()) > 0:
                # Score based on length and presence of construction-related keywords
                score = len(text.strip())
                construction_keywords = [
                    'concrete', 'steel', 'foundation', 'wall', 'floor', 'roof', 'beam', 'column', 
                    'footing', 'slab', 'dimension', 'length', 'width', 'height', 'thickness', 
                    'diameter', 'area', 'volume', 'square', 'cubic', 'linear', 'feet', 'inches',
                    'construction', 'building', 'structure', 'material', 'specification'
                ]
                keyword_count = sum(1 for keyword in construction_keywords if keyword.lower() in text.lower())
                score += keyword_count * 10
                
                if score > max_score:
                    max_score = score
                    best_text = text
        
        # Clean and format the text
        if best_text:
            # Remove excessive whitespace and clean up
            cleaned_text = re.sub(r'\s+', ' ', best_text.strip())
            
            return cleaned_text
    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return ""

# =================== AI COST ESTIMATION ===================
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

    # General category mapping
    categories = {
        "Concrete": ["CONCRETE", "FOUNDATION", "SLAB", "FOOTING", "STEM WALL", "REBAR", "CUBIC YARD", "PAVING"],
        "Masonry": ["BLOCK", "BRICK", "CMU", "MASONRY", "STONE", "MORTAR", "WALL"],
        "Electrical": ["BREAKER", "PANEL", "RECEPTACLE", "LIGHT", "SWITCH", "CONDUIT", "CIRCUIT", "AMP", "WATT", "BUS", "DIMMER", "FUSE", "TRANSFORMER"], #"AF", "AT", "KA", 
        "HVAC": ["DUCT", "AIR CONDITIONING", "AC", "AHU", "FCU", "FAN", "CHILLER", "HEATER", "VAV", "GRILLE", "DIFFUSER", "TEMPERATURE"],
        "Plumbing": ["PIPE", "DRAIN", "WATER CLOSET", "SINK", "TOILET", "VALVE", "PUMP", "HOT WATER", "COLD WATER", "SANITARY", "VENT"],
        "Finishes": ["PAINT", "DOOR", "WINDOW", "FLOOR", "CEILING", "CARPET", "TILE", "WALL COVERING", "MILLWORK", "TRIM"],
        "Fire Protection": ["SPRINKLER", "FIRE ALARM", "HYDRANT", "EXTINGUISHER", "SUPPRESSION", "DETECTOR", "SMOKE", "FIRE HOSE"],
        "Electrical Safety & Security": [
            "SMOKE DETECTOR", "HEAT DETECTOR", "MANUAL PULL STATION",
            "SPRINKLER", "EMERGENCY LIGHT", "EXIT SIGN", "SURGE PROTECTOR",
            "ACCESS CONTROL", "SECURITY CAMERA", "MOTION SENSOR", "CCTV",
            "INTRUSION ALARM", "CARD READER", "KEYPAD", "FIRE EXTINGUISHER",
            "FIRE SUPPRESSION", "SECURITY PANEL", "ALARM PANEL", "NOTIFICATION DEVICE"
        ],
        "Thermal & Moisture Protection": [
            "INSULATION", "VAPOR BARRIER", "ROOF MEMBRANE", "FLASHING", 
            "WATERPROOFING", "SEALANT", "CAULKING", "MOISTURE BARRIER", 
            "THERMAL BARRIER", "ROOF INSULATION", "SHEET MEMBRANE", 
            "TILE UNDERLAYMENT", "DAMP PROOFING", "AIR BARRIER", 
            "WEATHER BARRIER", "ROOFING FELT", "BITUMEN", "SPRAY FOAM"
        ],
        "Sitework": ["EXCAVATION", "GRADING", "BACKFILL", "ROADWAY", "CURB", "SIDEWALK", "ASPHALT", "DRAINAGE", "LANDSCAPING"],
        "Equipment": ["GENERATOR", "TRANSFORMER", "ELEVATOR", "CONVEYOR", "CONTROL", "AUTOMATION", "SECURITY", "CAMERA"],
        "Woods, Plastics & Composites": [
            "LUMBER", "PLYWOOD", "OSB", "TIMBER", "MDF", "PARTICLE BOARD", "LAMINATE", 
            "VINYL", "PVC", "PLASTIC PANEL", "COMPOSITE DECKING", "FIBERBOARD", "FIBERGLASS", 
            "RESIN", "ENGINEERED WOOD", "WOOD PANEL", "WOOD BEAM", "WOOD TRIM"
        ],
        "Other": []  # fallback
    }

    structured = defaultdict(dict)

    # --- Extract number + unit combos and count occurrences ---
    for cat, keywords in categories.items():
        for kw in keywords:
            # Match numbers before or after keyword (e.g., "250 AMP" or "AMP 250")
            matches = re.findall(r'(\d+(?:\.\d+)?)\s*' + re.escape(kw) + r'|' + re.escape(kw) + r'\s*(\d+(?:\.\d+)?)', text)
            counter = Counter()
            for m in matches:
                qty = m[0] or m[1]
                if qty:
                    item_str = f"{qty} {kw}"
                    counter[item_str] += 1
            for item_str, count in counter.items():
                structured[cat][item_str] = count

    # --- Count keyword-only occurrences ---
    for cat, keywords in categories.items():
        for kw in keywords:
            occurrences = len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
            # If already counted as number + keyword, subtract those
            num_kw_with_number = sum(1 for k in structured[cat].keys() if kw in k)
            qty = occurrences - num_kw_with_number
            if qty > 0:
                structured[cat][kw] = qty

    # --- Special Electrical rule ---
    if "Electrical" in structured:
        has_amp = any("AMP" in k for k in structured["Electrical"].keys())
        if has_amp:
            for kw in ["PANEL", "BUS"]:
                if kw in structured["Electrical"]:
                    del structured["Electrical"][kw]

    # --- Build AI-readable structured text ---
    parts = []
    for cat, items in structured.items():
        parts.append(f"{cat} Summary:")
        for item, qty in items.items():
            parts.append(f"- {item}: {qty}")
        parts.append("")  # spacing

    return "\n".join(parts).strip()

def extract_json_from_response(response_text: str):
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match: return match.group(1)
    match2 = re.search(r'\[.*\]', response_text, re.DOTALL)
    return match2.group(0) if match2 else None

def validate_data(data: list) -> bool:
    return isinstance(data, list) and len(data) > 0

def get_construction_jobs(cad_text):
    print(f"Starting construction jobs analysis...")
    
    # Preprocess the CAD text for better analysis
    processed_cad_text = preprocess_cad_text(cad_text)
    print(f"Preprocessed text ==> {processed_cad_text}")
    
    # Enhanced system prompt with specific instructions
    system_prompt = """You are a professional construction estimator with 20+ years of experience. Your task is to analyze CAD drawings and provide accurate cost estimates following these strict guidelines:

1. ACCURACY REQUIREMENTS:
   - Use 2024 MasterFormat standards (CSI codes)
   - Base costs on current US market rates (RSMeans data)
   - Consider regional variations (use national average if location unknown)
   - Provide realistic quantities based on drawing dimensions

2. REQUIRED OUTPUT FORMAT:
   - EXACTLY this JSON structure: [{"CSI code": "03 30 00", "Category": "Concrete", "Job Activity": "Cast-in-place Concrete Slab, 6-inch thick", "Quantity": 150, "Unit": "CY", "Rate": 125.50, "Material Cost": 12000, "Equipment Cost": 3000, "Labor Cost": 4500, "Total Cost": 19500}]
   - Job Activities must be clearly detailed including type, size, and method (e.g., "Cast-in-place Paving", "Electrical wire work - Conduit installation")
   - NO additional text, explanations, or comments
   - NO markdown formatting or code blocks
   - ALL fields must be present and properly formatted

3. COST ESTIMATION RULES:
   - Material Cost: 60-70% of total cost
   - Labor Cost: 25-35% of total cost  
   - Equipment Cost: 5-15% of total cost
   - Rates should reflect current market conditions
   - Quantities must be realistic based on drawing scale

4. VALIDATION CHECKLIST:
   - CSI codes must be valid MasterFormat codes
   - Quantities must be positive numbers
   - All costs must be positive numbers
   - Units must be standard construction units (SF, CY, LF, EA, etc.)
   - Total Cost = Material Cost + Equipment Cost + Labor Cost

If the CAD text is unclear or insufficient, provide estimates for common construction elements that would typically be found in the type of project indicated."""

    # Enhanced user prompt with better structure
    user_prompt = f"""Analyze this CAD drawing text and extract construction activities with accurate cost estimates:

CAD DRAWING TEXT:
{processed_cad_text}

REQUIRED ANALYSIS:
1. Identify all construction activities visible in the drawing
2. Determine appropriate quantities based on dimensions/text
3. Assign correct CSI codes from MasterFormat
4. Calculate realistic costs using 2024 US market rates
5. Ensure all mathematical relationships are correct

OUTPUT: Return ONLY a valid JSON array with the exact structure specified above. No additional text or formatting."""

    try:
        # First attempt with enhanced prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        print(f"Initial AI response received:: {response_text}")
        
        # Validate and potentially retry if response is poor
        validated_response = validate_and_improve_response(response_text, processed_cad_text)
        
        return validated_response
        
    except Exception as e:
        print(f"Error in get_construction_jobs: {e}")
        # Fallback to simpler prompt if main approach fails
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
        if "CSI code" in item:
            csi_code = str(item["CSI code"])
            if not re.match(r'^\d{2}\s\d{2}\s\d{2}$', csi_code):
                errors.append(f"Item {i} has invalid CSI code format: {csi_code}")
                score -= 5
    
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

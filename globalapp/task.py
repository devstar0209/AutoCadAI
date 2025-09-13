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

# Import ML model for enhanced accuracy
try:
    from ml_training import ConstructionCostPredictor
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False
    print("ML model not available. Using AI-only approach.")


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

# Initialize ML model if available
ml_predictor = None
if ML_MODEL_AVAILABLE:
    try:
        ml_predictor = ConstructionCostPredictor()
        # Try to load pre-trained models
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        if os.path.exists(model_dir):
            ml_predictor.load_models(model_dir)
            print("ML model loaded successfully!")
        else:
            print("No pre-trained ML model found. Using AI-only approach.")
            ml_predictor = None
    except Exception as e:
        print(f"Failed to load ML model: {e}")
        ml_predictor = None

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
    print(f"Entered enhanced OCR function: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found - {image_path}")
        return ""

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: cv2.imread failed - {image_path}")
            return ""

        # Enhanced image preprocessing for better OCR accuracy
        processed_text = ""
        
        # Method 1: Standard grayscale processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text1 = pytesseract.image_to_string(gray, config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-()[]{}:; ')
        
        # Method 2: Enhanced contrast and noise reduction
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)  # Increase contrast
        denoised = cv2.medianBlur(enhanced, 3)  # Reduce noise
        text2 = pytesseract.image_to_string(denoised, config='--psm 6')
        
        # Method 3: Adaptive thresholding for better text detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
                construction_keywords = ['concrete', 'steel', 'foundation', 'wall', 'floor', 'roof', 'beam', 'column', 'footing', 'slab', 'dimension', 'length', 'width', 'height', 'thickness', 'diameter', 'area', 'volume']
                keyword_count = sum(1 for keyword in construction_keywords if keyword.lower() in text.lower())
                score += keyword_count * 10
                
                if score > max_score:
                    max_score = score
                    best_text = text
        
        # Clean and format the text
        if best_text:
            # Remove excessive whitespace and clean up
            cleaned_text = re.sub(r'\s+', ' ', best_text.strip())
            # Remove common OCR artifacts
            cleaned_text = re.sub(r'[^\w\s.,-()\[\]{}:;]', '', cleaned_text)
            
            print(f"OCR extracted {len(cleaned_text)} characters")
            print("OCR text sample:", cleaned_text[:300])  # Print first 300 characters
            
            return cleaned_text
        else:
            print("No text extracted from image")
            return ""

    except Exception as e:
        print(f"Enhanced OCR Exception: {e}")
        # Fallback to simple OCR
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return pytesseract.image_to_string(gray)
        except:
        return ""
def preprocess_cad_text(cad_text):
    """Preprocess CAD text to improve AI analysis accuracy"""
    if not cad_text or len(cad_text.strip()) < 10:
        return "CAD drawing contains minimal text. Please provide estimates for common construction elements: foundation, walls, roof, and basic finishes."
    
    # Clean and structure the text
    cleaned_text = re.sub(r'\s+', ' ', cad_text.strip())
    
    # Check if text contains construction-related content
    construction_indicators = [
        'concrete', 'steel', 'foundation', 'wall', 'floor', 'roof', 'beam', 'column', 
        'footing', 'slab', 'dimension', 'length', 'width', 'height', 'thickness', 
        'diameter', 'area', 'volume', 'square', 'cubic', 'linear', 'feet', 'inches',
        'construction', 'building', 'structure', 'material', 'specification'
    ]
    
    has_construction_content = any(indicator.lower() in cleaned_text.lower() for indicator in construction_indicators)
    
    if not has_construction_content:
        return f"CAD drawing text: {cleaned_text}\n\nNote: Limited construction-specific information detected. Please provide estimates for standard building components based on typical construction practices."
    
    return cleaned_text

def get_construction_jobs_hybrid(cad_text, image_path=None):
    """
    Hybrid approach combining ML predictions with AI analysis for maximum accuracy.
    """
    print("Starting hybrid construction jobs analysis...")
    
    # Preprocess the CAD text for better analysis
    processed_cad_text = preprocess_cad_text(cad_text)
    
    # Get ML predictions if model is available and image path is provided
    ml_predictions = None
    if ml_predictor and image_path and os.path.exists(image_path):
        try:
            print("Getting ML predictions...")
            ml_predictions = ml_predictor.predict_costs(image_path)
            print(f"ML predictions: {ml_predictions}")
        except Exception as e:
            print(f"ML prediction failed: {e}")
            ml_predictions = None
    
    # Get AI analysis
    ai_response = get_construction_jobs_ai_only(processed_cad_text)
    
    # Combine ML and AI results if both are available
    if ml_predictions and ai_response:
        return combine_ml_ai_results(ai_response, ml_predictions, processed_cad_text)
    elif ml_predictions:
        return convert_ml_predictions_to_json(ml_predictions)
    else:
        return ai_response

def get_construction_jobs_ai_only(cad_text):
    """Original AI-only approach for construction jobs analysis."""
    print("Starting AI-only construction jobs analysis...")
    
    # Preprocess the CAD text for better analysis
    processed_cad_text = preprocess_cad_text(cad_text)
    
    # Enhanced system prompt with specific instructions
    system_prompt = """You are a professional construction estimator with 20+ years of experience. Your task is to analyze CAD drawings and provide accurate cost estimates following these strict guidelines:

1. ACCURACY REQUIREMENTS:
   - Use 2024 MasterFormat standards (CSI codes)
   - Base costs on current US market rates (RSMeans data)
   - Consider regional variations (use national average if location unknown)
   - Provide realistic quantities based on drawing dimensions

2. REQUIRED OUTPUT FORMAT:
   - EXACTLY this JSON structure: [{"CSI code": "03 30 00", "Category": "Concrete", "Job Activity": "Foundation", "Quantity": 150, "Unit": "CY", "Rate": 125.50, "Material Cost": 12000, "Equipment Cost": 3000, "Labor Cost": 4500, "Total Cost": 19500}]
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
        print("Initial AI response received")
        
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
                return response_text
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
        print("Improved response generated based on feedback")
        return improved_response
        
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
        # Return minimal valid response
        return '[{"CSI code": "00 00 00", "Category": "General", "Job Activity": "Project Setup", "Quantity": 1, "Unit": "EA", "Rate": 1000, "Material Cost": 500, "Equipment Cost": 200, "Labor Cost": 300, "Total Cost": 1000}]'

def combine_ml_ai_results(ai_response, ml_predictions, cad_text):
    """
    Combine ML predictions with AI analysis for enhanced accuracy.
    """
    try:
        # Parse AI response
        clean_json = extract_json_from_response(ai_response)
        if not clean_json:
            return convert_ml_predictions_to_json(ml_predictions)
        
        ai_data = json.loads(clean_json)
        if not isinstance(ai_data, list) or len(ai_data) == 0:
            return convert_ml_predictions_to_json(ml_predictions)
        
        # Use ML predictions to validate and adjust AI results
        ml_total = ml_predictions.get('total_cost', 0)
        ai_total = sum(float(item.get('Total Cost', 0)) for item in ai_data)
        
        # Calculate adjustment factor
        if ai_total > 0 and ml_total > 0:
            adjustment_factor = ml_total / ai_total
            print(f"Adjusting AI results by factor: {adjustment_factor:.3f}")
            
            # Apply adjustment to AI results
            for item in ai_data:
                for cost_field in ['Material Cost', 'Equipment Cost', 'Labor Cost', 'Total Cost']:
                    if cost_field in item:
                        original_cost = float(item[cost_field])
                        adjusted_cost = original_cost * adjustment_factor
                        item[cost_field] = round(adjusted_cost, 2)
                
                # Recalculate rate
                if 'Quantity' in item and 'Total Cost' in item:
                    quantity = float(item['Quantity'])
                    if quantity > 0:
                        item['Rate'] = round(float(item['Total Cost']) / quantity, 2)
        
        return json.dumps(ai_data)
        
    except Exception as e:
        print(f"Error combining ML and AI results: {e}")
        return convert_ml_predictions_to_json(ml_predictions)

def convert_ml_predictions_to_json(ml_predictions):
    """
    Convert ML predictions to the expected JSON format.
    """
    try:
        # Create a basic construction job entry from ML predictions
        ml_data = [{
            "CSI code": "00 00 00",
            "Category": "General Construction",
            "Job Activity": "Complete Project",
            "Quantity": 1,
            "Unit": "EA",
            "Rate": ml_predictions.get('total_cost', 0),
            "Material Cost": ml_predictions.get('material_cost', 0),
            "Equipment Cost": ml_predictions.get('equipment_cost', 0),
            "Labor Cost": ml_predictions.get('labor_cost', 0),
            "Total Cost": ml_predictions.get('total_cost', 0)
        }]
        
        return json.dumps(ml_data)
        
    except Exception as e:
        print(f"Error converting ML predictions: {e}")
        return '[{"CSI code": "00 00 00", "Category": "General", "Job Activity": "Project Setup", "Quantity": 1, "Unit": "EA", "Rate": 1000, "Material Cost": 500, "Equipment Cost": 200, "Labor Cost": 300, "Total Cost": 1000}]'

# Keep the original function name for backward compatibility
def get_construction_jobs(cad_text, image_path=None):
    """
    Main function for construction jobs analysis.
    Uses hybrid approach if ML model is available, otherwise falls back to AI-only.
    """
    if ml_predictor and image_path:
        return get_construction_jobs_hybrid(cad_text, image_path)
    else:
        return get_construction_jobs_ai_only(cad_text)

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
    job_data = get_construction_jobs(cad_text, image_path)
    
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

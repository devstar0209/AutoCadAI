import os
import re
import json
import cv2
import pytesseract
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from difflib import SequenceMatcher


# -----------------------------
# CONFIGURATION
# -----------------------------
PDF_FOLDER = "pdfs"           # folder containing drawing PDFs
EXCEL_FOLDER = "excels"       # folder containing Excel cost files
OUTPUT_FILE = "fine_tune_dataset.jsonl"


MAX_WORKERS = 4

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
        "panelboard", "circuit", "conduit", "breaker", "wire", "receptacle", 
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

def get_page_count(pdf_file):
    reader = PdfReader(pdf_file)
    return len(reader.pages)

def convert_pdf_page_to_image(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(pdf_path, dpi=200, first_page=page_number, last_page=page_number, poppler_path="/usr/bin")
    if not images: return ""
    directory = os.path.dirname(pdf_path)
    image_path = os.path.join(directory, f"page_{page_number}.png")
    images[0].save(image_path, "PNG")
    return image_path

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

def extract_pdf_text(pdf_path: str) -> str:
    total_pages = get_page_count(pdf_path)
    all_texts = [""] * total_pages

    def process_page(page_num):
        img_path = convert_pdf_page_to_image(pdf_path, page_num)
        if img_path:
            all_texts[page_num-1] = extract_text_from_image(img_path)
        

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_page, range(1, total_pages + 1))

    return "\n".join(all_texts)

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

# -----------------------------
# 2. READ EXCEL COST DATA
# -----------------------------
    
def read_excel_costs(excel_path):
    SHEETS_TO_USE = ["Summary", "Main Building", "Work Scope", "Labor Rates"]

    ext = os.path.splitext(excel_path)[1].lower()
    if ext == ".xls":
        engine = "xlrd"
    elif ext == ".xlsx":
        engine = "openpyxl"
    else:
        raise ValueError(f"Unsupported Excel format: {ext}")

    summary_data = []
    detail_items = []

    xls = pd.ExcelFile(excel_path, engine=engine)

    for sheet in SHEETS_TO_USE:
        if sheet not in xls.sheet_names:
            continue

        # Load without header to detect start row
        df_raw = pd.read_excel(xls, sheet_name=sheet, header=None, engine=engine)

        # Find header row dynamically (look for keywords)
        header_row = None
        for i, row in df_raw.iterrows():
            row_lower = [str(c).lower() for c in row.values if pd.notna(c)]
            if any(key in row_lower for key in ["description", "category", "activity", "unit", "total cost"]):
                header_row = i
                break

        if header_row is None:
            continue  # Skip sheet if no valid header found

        # Reload with detected header
        df = pd.read_excel(xls, sheet_name=sheet, header=header_row, engine=engine)

        if sheet.lower() == "summary":
            for _, row in df.iterrows():
                desc = str(row.get("Description", "")).strip()
                cost = row.get("Total Cost") or row.get("Total") or 0

                if desc and pd.notna(cost) and float(cost) > 0:
                    summary_data.append({
                        "Div": row.get("Div", ""),
                        "Category": desc
                    })
        else:
            for _, row in df.iterrows():
                description = str(row.get("DESCRIPTION", ""))
                div = row.get("Div", "")
                unit = str(row.get("Unit", "")).strip().lower()
                if div != "" and unit != "":
                    try:
                        detail_items.append({
                            "Div": div,
                            "Job Activity": description,
                            "Quantity": str(row.get("Quant.", 0)),
                            "Unit": unit,
                            "L.hrs": round(float(row.get("L.Hrs.", 0)),1),
                            "E.hrs": round(float(row.get("E.Hrs.", 0)), 1),
                        })
                    except Exception as e:
                        print(f"Error processing row in {sheet}: {e}")
                        continue

    return summary_data, detail_items

def clean_ocr_text(text: str) -> str:
    # 1. Remove backslashes, unicode escapes, and control chars
    text = text.encode("utf-8", "ignore").decode()
    text = re.sub(r'\\[uU]\w{4}', ' ', text)
    text = re.sub(r'[^ -~\n]', ' ', text)  # keep printable chars

    # 2. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # # 4. Split into potential sentences or sections
    # chunks = re.split(r'(?<=\.)(?=\s[A-Z])|(?=### )', text)
    # chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    
    return text

# ----------------------------------------
# 🔸 Utility: text cleanup
# ----------------------------------------
def normalize_text(s):
    """Normalize for OCR fuzziness."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)   # remove punctuation
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# ----------------------------------------
# 🔸 Utility: fuzzy similarity
# ----------------------------------------
def fuzzy_ratio(a, b):
    """Basic fuzzy similarity score."""
    return SequenceMatcher(None, a, b).ratio()

# -----------------------------
# 3. GENERATE JSONL ENTRIES
# -----------------------------
dataset_entries = []

# Iterate through all projects
for pdf_file in os.listdir(PDF_FOLDER):
    if not pdf_file.lower().endswith(".pdf"):
        continue
    project_name = os.path.splitext(pdf_file)[0]
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    txt_path = pdf_path.replace(".pdf", ".txt")
    json_path = pdf_path.replace(".pdf", ".json")

    # Corresponding Excel
    excel_path = os.path.join(EXCEL_FOLDER, project_name + ".xls")
    if not os.path.exists(excel_path):
        excel_path = os.path.join(EXCEL_FOLDER, project_name + ".xlsx")
    if not os.path.exists(excel_path):
        print(f"⚠️ Excel file not found for {project_name}")
        continue

    summary_data, detail_items = read_excel_costs(excel_path)

    if not os.path.exists(txt_path):
        print("File does not exist")
        pdf_text = extract_pdf_text(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(pdf_text)

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
    
    # Get unique divisions from summary
    divisions = {
        (s["Div"], s["Category"])
        for s in summary_data
        if s.get("Div") and s.get("Category")
    }
    details_clean = []
    for div, category in divisions:
        # Filter detail items for this division
        details_div = [d for d in detail_items if d.get("Div", "") == div]
        
        try:
            ocr_text = combined_category_text[category]
            
            if not details_div:
                continue 

            user_content = f"OCR text: {ocr_text}. Extract all job activities for category-{category} with unit, quantity, labor working hours and equipment working hours in JSON.\n"
            
            details_clean = []
            for d in details_div:
                d_clean = {k: v for k, v in d.items() if k != "Div"} # Remove "Div" field from details
                d_clean["Category"] = category
                details_clean.append(d_clean)
            
            assistant_content = json.dumps(details_clean)

            dataset_entries.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            })
        except Exception as e:
            print(f"Error processing division {div}, category {category}: {e}")
            continue


# Write JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in dataset_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Fine-tuning dataset created with {len(dataset_entries)} entries: {OUTPUT_FILE}")

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

        total_cost = 0

        if sheet.lower() == "summary":
            for _, row in df.iterrows():
                desc = str(row.get("Description", "")).strip()
                cost = row.get("Total Cost") or row.get("Total") or 0

                if desc and pd.notna(cost) and float(cost) > 0:
                    summary_data.append({
                        "Div": row.get("Div"),
                        "Category": desc,
                        "TotalCost": round(float(cost), 2)
                    })
        else:
            for _, row in df.iterrows():
                subtotal = row.get("Subtotal") or row.get("Sub.Cost") or 0
                total = row.get("Total") or row.get("Cost") or 0
                description = str(row.get("DESCRIPTION", ""))
                if description == "Total Project Cost":
                    total_cost = total
                    continue
                if (pd.isna(subtotal) or float(subtotal) <= 0) and (pd.isna(total) or float(total) <= 0):
                    continue
                if (float(total) > 0) and (pd.isna(subtotal) or float(subtotal) <= 0):
                    subtotal = total

                detail_items.append({
                    "Div": row.get("Div"),
                    "Job Activity": description,
                    "Quantity": str(row.get("Quant.", "")),
                    "Unit": str(row.get("Unit", "")),
                    "Rate": round(float(row.get("Rate", 0) or 0), 2),
                    "Material Cost": round(float(row.get("T.Mat", 0) or 0), 2),
                    "Labor Cost": round(float(row.get("T.Labor", 0) or 0), 2),
                    "Equipment Cost": round(float(row.get("T.Euip", 0) or 0), 2),
                    "Sub Markups": round(float(row.get("Markups", 0) or 0), 2),
                    "Subtotal Cost": round(float(subtotal), 2),
                })

    return summary_data, detail_items, total_cost

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

def smart_chunk_text(ocr_text: str, max_chunk_len=1200, min_chunk_len=150):
    """
    Splits OCR text into logical 'job activity' chunks using regex patterns.
    - Groups related lines until likely new activity starts
    - Keeps chunks semantically complete for fine-tuning
    """

    lines = [ln.strip() for ln in ocr_text.split("\n") if ln.strip()]
    chunks = []
    current_chunk = ""

    job_start_pattern = re.compile(
        r"^([A-Z][a-zA-Z ]{3,}|Protect|Install|Provide|Remove|Excavate|Construct|Repair|Clean|Paint)",
        re.IGNORECASE,
    )
    has_cost_pattern = re.compile(r"\b\d{2,}\.\d{2}\b")
    has_unit_pattern = re.compile(r"\b(\d+(\.\d+)?)\s?(ea|lf|sf|ft|allow|each)\b", re.IGNORECASE)

    for line in lines:
        # Check if this line is likely a new job activity
        new_job = bool(job_start_pattern.search(line) and (has_cost_pattern.search(line) or has_unit_pattern.search(line)))

        if new_job and len(current_chunk) >= min_chunk_len:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            if len(current_chunk) + len(line) < max_chunk_len:
                current_chunk += " " + line
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line

    # Final chunk
    if len(current_chunk.strip()) > 0:
        chunks.append(current_chunk.strip())

    # Merge tiny chunks with previous ones
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < min_chunk_len:
            merged[-1] += " " + chunk
        else:
            merged.append(chunk)

    return merged
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


# ----------------------------------------
# 🔸 Advanced fuzzy+keyword matching
# ----------------------------------------
def match_jobs_to_chunk_ocr_tolerant(chunk, assistant_json,
                                     fuzzy_threshold=0.55,
                                     keyword_overlap_threshold=0.25):
    """
    Return only those jobs whose 'Job Activity' roughly appears
    in the chunk, tolerant to OCR distortions and partial matches.
    """
    matched = []
    chunk_norm = normalize_text(chunk)
    chunk_tokens = set(chunk_norm.split())

    for job in assistant_json:
        job_name = job["Job Activity"]
        if not job_name:
            continue

        job_norm = normalize_text(job_name)
        job_tokens = set(job_norm.split())

        # 1️⃣ Fuzzy match on full strings
        sim = fuzzy_ratio(job_norm, chunk_norm)

        # 2️⃣ Token overlap score (intersection over smaller set)
        overlap = len(chunk_tokens & job_tokens) / max(1, len(job_tokens))

        # 3️⃣ Decide match
        if sim >= fuzzy_threshold or overlap >= keyword_overlap_threshold:
            matched.append(job)

    return matched
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
    pdf_text = extract_pdf_text(pdf_path)

    # Corresponding Excel
    excel_path = os.path.join(EXCEL_FOLDER, project_name + ".xls")
    if not os.path.exists(excel_path):
        excel_path = os.path.join(EXCEL_FOLDER, project_name + ".xlsx")
    if not os.path.exists(excel_path):
        print(f"⚠️ Excel file not found for {project_name}")
        continue

    summary_data, detail_items, total_cost = read_excel_costs(excel_path)

    dataset_entries = []

    user_content = f"Analyze the following drawing OCR text and organize them by trade category and estimate costs. Drawing OCR Text:\n"

    # Get unique divisions from summary
    divisions = set(s["Div"] for s in summary_data if s.get("Div"))
    details_clean = []
    for div in divisions:
        # Filter detail items for this division
        details_div = [d for d in detail_items if d.get("Div") == div]
        # summary_div = [d for d in summary_data if d.get("Div") == div]

        if not details_div:
            continue  # skip empty div

        # Remove "Div" field from details
        details_clean = []
        for d in details_div:
            d_clean = {k: v for k, v in d.items() if k != "Div"}
            # Assign category based on summary Description of this Div
            # Pick first matching summary description for this div
            summary_descs = [s["Category"] for s in summary_data if s.get("Div") == div]
            d_clean["Category"] = summary_descs[0] if summary_descs else ""
            details_clean.append(d_clean)
        
        assistant_content = json.dumps(details_clean)

        dataset_entries.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    # clean_text = clean_ocr_text(pdf_text)
    # chunks = smart_chunk_text(clean_text)

    # for chunk in chunks:
    #     if len(chunk) < 120:
    #         continue

    #     matched_jobs = match_jobs_to_chunk_ocr_tolerant(chunk, details_clean)

    #     if not matched_jobs:
    #         continue  # skip irrelevant chunk

    #     entry = {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": f"Extract all job activities, quantities, costs, and categories from this OCR text:\n{chunk}"
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": json.dumps(matched_jobs, ensure_ascii=False)
    #             }
    #         ]
    #     }
        # dataset_entries.append(entry)


# Write JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in dataset_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Fine-tuning dataset created with {len(dataset_entries)} entries: {OUTPUT_FILE}")

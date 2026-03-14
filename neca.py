import pdfplumber
import re
import json

PDF_FILE = "NECA - Electrical Installation Labor Productivity Output.pdf"
OUTPUT_FILE = "neca.json"

data = []

current_div = None
current_category = None
current_csi = None
current_group = None


def parse_section(text):
    m = re.search(r"Division\s+(\d+)\s*—\s*(.+)", text)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, None


def parse_csi(text):
    m = re.match(r"(\d{2}\s\d{2}\s\d{2})\s*:\s*(.+)", text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def is_number(v):
    try:
        float(v)
        return True
    except:
        return False


print("Starting PDF parsing...\n")

with pdfplumber.open(PDF_FILE) as pdf:

    print(f"Total pages: {len(pdf.pages)}\n")

    for page_index, page in enumerate(pdf.pages):

        if page_index < 256:
            continue

        print(f"\n=========== PAGE {page_index + 1} ===========")

        page_text = page.extract_text()

        # STEP 1: Detect section header
        if page_text:
            for line in page_text.split("\n"):

                if "Division" in line and "Section" in line:

                    print(f"Found Section Header: {line}")

                    div, cat = parse_section(line)

                    if div:
                        current_div = div
                        current_category = cat

                        print(f"Parsed Division: {current_div}")
                        print(f"Parsed Category: {current_category}")

        tables = page.extract_tables()

        if not tables:
            print("No tables found on this page.")
            continue

        print(f"Tables found: {len(tables)}")

        # STEP 2: Parse tables
        for table_index, table in enumerate(tables):
            print(f"\n--- Processing Table {table_index + 1} ---")

            for row_index, row in enumerate(table):
                print(f"\nRow {row_index}: {row}")

                if not row:
                    print("Skipping empty row")
                    continue

                # Join multi-line cells into one string
                row = [cell.replace("\n", " ").strip() if cell else "" for cell in row]

                desc = row[1] if len(row) > 1 else None

                if not desc:
                    print("No description column, skipping")
                    continue

                print(f"Description detected: {desc}")

                # STEP 3: Detect CSI
                csi, csi_name = parse_csi(desc)

                if csi:
                    current_csi = csi
                    current_group = None

                    print(f"CSI detected: {current_csi}")
                    print(f"CSI Title: {csi_name}")

                    continue

                # STEP 4: Detect group
                normal = row[3] if len(row) > 3 else None
                difficult = row[4] if len(row) > 4 else None
                very_difficult = row[5] if len(row) > 5 else None

                if not normal and not difficult and not very_difficult and "Note:" not in desc:
                    current_group = desc
                    print(f"Group detected: {current_group}")
                    continue

                # STEP 5: Parse productivity values
                try:
                    normal_val = float(normal)
                    difficult_val = float(difficult)
                    very_difficult_val = float(very_difficult)
                except:
                    print("Values not numeric, skipping row")
                    continue

                unit = row[7] if len(row) > 7 else None

                item_name = desc
                if current_group:
                    item_name = f"{desc}, {current_group}"

                record = {
                    "div": current_div,
                    "category": current_category,
                    "csi": current_csi,
                    "item": item_name,
                    "labor_productivity_rate": {
                        "normal": normal_val,
                        "difficult": difficult_val,
                        "very_difficult": very_difficult_val
                    },
                    "unit": unit
                }

                print("Item parsed successfully:")
                print(record)

                data.append(record)

                # Save JSON incrementally
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(data, f, indent=2)


print("\n=========== FINISHED PARSING ===========")
print(f"Total items extracted: {len(data)}")

# STEP 6: Save JSON


print(f"\nJSON saved to: {OUTPUT_FILE}")
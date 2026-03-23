import pandas as pd
import json
import os

file_path = "excels/hollywood.xls"
target_sheet = "Option 1"
year = '2025'

# --- Step 1: Choose engine automatically ---
ext = os.path.splitext(file_path)[1].lower()
if ext == ".xls":
    engine = "xlrd"
else:
    engine = "openpyxl"

# --- Step 2: Load raw sheet ---
raw_df = pd.read_excel(file_path, sheet_name=target_sheet, header=None, dtype=str, engine=engine)

# --- Step 3: Detect header row ---
header_row_index = None
for i, row in raw_df.iterrows():
    row_str = row.astype(str).str.lower().tolist()
    if any("description" in cell for cell in row_str):
        header_row_index = i
        break
print(f"🔍 Detected header row at index: {header_row_index}")
if header_row_index is None:
    raise Exception("❌ Header row not found")

# --- Step 4: Reload with correct header ---
df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_row_index, dtype=str, engine=engine)

# --- Step 5: Clean column names ---
df.columns = df.columns.str.strip()

# --- Step 7: Filter rows ---
first_col = df.columns[0]

df = df[
    # (df[first_col].astype(str).str.strip().str.lower() == "show") &
    (df["Div"].notna()) &
    (df["DESCRIPTION"].notna()) &
    (df["Unit"].notna())
]

# --- Step 8: Build JSON ---
data = []

print(f"Processing {len(df)} rows from sheet '{target_sheet}'...")

for _, row in df.iterrows():
    item = {
        "Div": str(row["Div"]),
        "description": str(row["DESCRIPTION"]),
        "Unit": str(row["Unit"]).upper(),
        "unit_material_cost": float(pd.to_numeric(row["U/Cost"], errors="coerce") or 0),
        "labor_hours_per_unit": float(pd.to_numeric(row["MH/Unit"], errors="coerce") or 0),
        "unit_labor_cost": float(pd.to_numeric(row["L.Rate"], errors="coerce") or 0),
        "equipment_hours_per_unit": float(pd.to_numeric(row["H/Unit"], errors="coerce") or 0),
        "unit_equipment_cost": float(pd.to_numeric(row["E.Rate"], errors="coerce") or 0),
        "year": year
    }

    for k, v in item.items():
        if pd.isna(v):
            item[k] = 0

    data.append(item)

# --- Step 9: Save ---
with open("parsed_data.json", "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ Parsed {len(data)} rows from {target_sheet}")
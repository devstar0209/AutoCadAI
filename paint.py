import json
import re

INPUT_FILE = "painting_data.txt"
OUTPUT_JSON = "painting_data.json"

# Load existing JSON if exists
try:
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

unit = lines[0]  # first line = SF or LF
category = None
current_application = ""
current_material = ""
material_id = None

for line in lines[1:]:
    # Detect material line
    if "material" in line.lower():
        current_material = line
        # Extract all material IDs in the line (handle combined materials like #3 + #9)
        material_ids = [int(m) for m in re.findall(r"#(\d+)", current_material)]
        material_id = material_ids if material_ids else None
        continue

    # Detect application line (not starting with Slow/Medium/Fast, not a material line)
    if not any(line.startswith(prefix) for prefix in ["Slow", "Medium", "Fast"]) and "material" not in line.lower():
        current_application = line
        # If category not set, first application is the category
        if category is None:
            category = line
        continue

    # Speed variant numeric line
    parts = line.split()
    # if len(parts) != 10:
    #     print(f"Skipping malformed line: {line}")
    #     continue

    speed_label = parts[0]
    new_record = {
        "category": category,
        "unit": unit,
        "application": current_application,
        "material": current_material,
        "speed": speed_label,
        "labor_units_per_hour": float(parts[1]) if parts[1] != "--" else None,
        "material_coverage_per_unit": float(parts[2]) if parts[2] != "--" else None,
        "material_cost_per_unit": float(parts[3]) if parts[3] != "--" else None,
        "labor_cost_per_100unit": float(parts[4]) if parts[4] != "--" else None,
        "labor_burden_per_100unit": float(parts[5]) if parts[5] != "--" else None,
        "material_cost_per_100unit": float(parts[6]) if parts[6] != "--" else None,
        "overhead_per_100unit": float(parts[7]) if parts[7] != "--" else None,
        "profit_per_100unit": float(parts[8]) if parts[8] != "--" else None,
        "total_price_per_100unit": float(parts[9]) if parts[9] != "--" else None
    }

    # Update existing entry if same category + application + material + speed
    updated = False
    for i, record in enumerate(data):
        if (record.get("category") == category and
            record.get("application") == current_application and
            record.get("material") == current_material and
            record.get("speed") == speed_label):
            data[i] = new_record
            updated = True
            break
    if not updated:
        data.append(new_record)

# Save back to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Total entries now: {len(data)}")
import re
import json
import os
from typing import List, Dict

# ==============================
# Flexible OCR line regex
# ==============================
ITEM_LINE = re.compile(
    r"""
    (?P<desc>.+?)                      # Size / description
    \s+(?P<hour_type>P1|ER|SL|SK|SN|CF|BE|S2)@(?P<hours>[\d.]+)   # Labor or Equipment hours
    \s+(?P<unit>\w+)                   # Unit
    (?:\s+(?P<material>[\d.,]+))?      # OPTIONAL Material cost
    \s*(?:[—-]\s*)?
    \s+(?P<labor>[\d.,]+)              # Labor cost
    (?:\s+(?P<equipment>[\d.,]+))?     # OPTIONAL equipment cost
    \s*(?:[—-]\s*)?                    # OPTIONAL dash
    (?P<total>[\d.,]+)                 # Total
    $
    """,
    re.VERBOSE
)

# ==============================
# Helpers
# ==============================
def clean_money(val: str | None) -> float | None:
    if val is None:
        return None
    return float(val.replace(",", ""))

# ==============================
# OCR Parsing Logic
# ==============================
def parse_ocr_text(ocr_text: str) -> List[Dict]:
    results = []
    current_section = None
    item_buffer = []

    for line in ocr_text.splitlines():
        line = line.strip()

        if not line:
            item_buffer = []
            continue

        # First non-empty line is section header
        if current_section is None:
            current_section = line
            continue

        # Accumulate description lines
        if "@" not in line:
            item_buffer.append(line)
            continue

        # Combine buffered description + cost line
        item_text = " ".join(item_buffer + [line])
        item_buffer = []

        match = ITEM_LINE.search(item_text)
        if not match:
            continue

        hour_type = match.group("hour_type")

        labor_hours = None
        equipment_hours = None

        # if hour_type == "P1" or hour_type == "SL" or hour_type == "SK":
        #     labor_hours = float(match.group("hours"))
        # elif hour_type == "ER":
        #     equipment_hours = float(match.group("hours"))

        results.append({
            "item": f"{match.group('desc').strip()} {current_section}",
            "unit": match.group("unit"),
            "labor_hours": float(match.group("hours")),
            # "equipment_hours": equipment_hours,
            "material_cost": clean_money(match.group("material")),
            "labor_cost": clean_money(match.group("labor")),
            "equipment_cost": clean_money(match.group("equipment")),
            "total_cost": clean_money(match.group("total")),
        })

    return results

# ==============================
# JSON Helpers
# ==============================
def load_existing_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

# ==============================
# Main
# ==============================
def main():
    ocr_file = "hvac.txt"
    output_json = "hvac.json"

    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    new_items = parse_ocr_text(ocr_text)
    existing_items = load_existing_json(output_json)

    existing_keys = {i["item"] for i in existing_items}
    new_items = [i for i in new_items if i["item"] not in existing_keys]

    existing_items.extend(new_items)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(existing_items, f, indent=2, ensure_ascii=False)

    print(f"Added {len(new_items)} items")
    print(f"Total items: {len(existing_items)}")

# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    main()
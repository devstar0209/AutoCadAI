import re
import json
import os
from typing import List, Dict

# Regex to match a line containing costs (P1@...)
ITEM_LINE = re.compile(
    r"""
    (?P<desc>.+?)                # Description before P1@
    \s+P1@(?P<hours>[\d.]+)     # Labor hours
    \s+(?P<unit>\w+)             # Unit (Ea, LF, etc.)
    \s+(?P<material>[\d.,]+)    # Material cost
    \s+(?P<labor>[\d.,]+)       # Labor cost
    \s+[—-]\s+(?P<total>[\d.,]+) # Total cost
    """,
    re.VERBOSE
)

def clean_money(val: str) -> float:
    return float(val.replace(",", ""))

def parse_ocr_text(ocr_text: str) -> List[Dict]:
    results = []
    current_section = None
    item_buffer = []  # Collect multi-line item descriptions

    for line in ocr_text.splitlines():
        line = line.strip()
        if not line:
            # Reset item buffer on blank lines
            item_buffer = []
            continue

        # If no section yet, first line = section header
        if current_section is None:
            current_section = line
            continue

        # Add line to buffer until we find a P1@ line
        if "P1@" not in line:
            item_buffer.append(line)
            continue

        # Line with P1@ found → combine buffer + this line
        item_text = " ".join(l.strip(",") for l in item_buffer + [line])
        item_buffer = []  # reset buffer

        match = ITEM_LINE.search(item_text)
        if match:
            full_item_name = f"{current_section} {match.group('desc').strip()}"
            results.append({
                "item": full_item_name,
                "hours": float(match.group("hours")),
                "unit": match.group("unit"),
                "material_cost": clean_money(match.group("material")),
                "labor_cost": clean_money(match.group("labor")),
                "equipment_cost": 0.0,
                "total_cost": clean_money(match.group("total")),
            })
        else:
            # If no match, just skip (could log)
            continue

    return results

# JSON append logic
def load_existing_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def main():
    ocr_file = "hvac.txt"
    output_json = "hvac.json"

    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    new_items = parse_ocr_text(ocr_text)
    existing_items = load_existing_json(output_json)

    # Prevent duplicates
    existing_keys = {i["item"] for i in existing_items}
    new_items = [i for i in new_items if i["item"] not in existing_keys]

    existing_items.extend(new_items)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(existing_items, f, indent=2, ensure_ascii=False)

    print(f"Added {len(new_items)} items")
    print(f"Total items: {len(existing_items)}")

if __name__ == "__main__":
    main()
import re
import json
import os
from typing import List, Dict

# ============================
# Regex Patterns
# ============================

ONE_LINE_ROW = re.compile(
    r"""
    (?P<size>.+?)                   # Everything before P1@
    \s+P1@(?P<hours>[\d.]+)        # Labor hours
    \s+(?P<unit>\w+)                # Unit can be Ea, LF, SF, etc.
    \s+(?P<material>[\d.,]+)       # Material cost
    \s+(?P<labor>[\d.,]+)          # Labor cost
    \s+[—-]\s+(?P<total>[\d.,]+)   # Total cost
    """,
    re.VERBOSE
)

INLINE_HEATER = re.compile(
    r"""
    (?P<capacity>\d+\s+gallon),\s*
    (?P<kw>\d+\s*kw)
    \s+P1@(?P<hours>[\d.]+)
    \s+(?P<unit>\w+)
    \s+(?P<material>[\d.,]+)
    \s+(?P<labor>[\d.,]+)
    \s+[—-]\s+(?P<total>[\d.,]+)
    """,
    re.VERBOSE | re.IGNORECASE
)

SPEC_ROW = re.compile(
    r"""
    (?P<spec>\d+(\.\d+)?\s*KW\/\d+V)
    \s+P1@(?P<hours>[\d.]+)
    \s+(?P<unit>\w+)
    \s+(?P<material>[\d.,]+)
    \s+(?P<labor>[\d.,]+)
    \s+[—-]\s+(?P<total>[\d.,]+)
    """,
    re.VERBOSE | re.IGNORECASE
)

# ============================
# Helpers
# ============================

def clean_money(val: str) -> float:
    return float(val.replace(",", ""))

def is_section_header(line: str) -> bool:
    """Detect section header lines."""
    if "P1@" in line:
        return False
    if re.search(r"\b(Ea|LF|SF|HR)\b", line):
        return False
    if re.search(r"[—-]\s*\d", line):
        return False
    return len(line.split()) >= 2

# ============================
# OCR Parser
# ============================

def parse_ocr_text(ocr_text: str) -> List[Dict]:
    results = []
    current_section = None
    multi_line_buffer = []

    for raw_line in ocr_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.replace("—", "-")

        # -----------------------------
        # Section header (first line of a section)
        # -----------------------------
        if is_section_header(line):
            current_section = line
            multi_line_buffer = []
            continue

        # -----------------------------
        # Multi-line description (before P1@)
        # -----------------------------
        if 'P1@' not in line:
            multi_line_buffer.append(line.strip(','))
            continue

        # -----------------------------
        # Process ONE_LINE_ROW
        # -----------------------------
        row = ONE_LINE_ROW.search(line)
        if row and current_section:
            # Item name = section + multi-line buffer + size/spec
            parts = [current_section] + multi_line_buffer + [row.group('size').strip()]
            full_item_name = ' '.join(parts)
            multi_line_buffer = []

            results.append({
                "item": full_item_name.strip(),
                "hours": float(row.group("hours")),
                "unit": row.group("unit"),
                "material_cost": clean_money(row.group("material")),
                "labor_cost": clean_money(row.group("labor")),
                "equipment_cost": 0.0,
                "total_cost": clean_money(row.group("total")),
            })
            continue

        # -----------------------------
        # INLINE_HEATER (e.g., commercial water heaters)
        # -----------------------------
        inline = INLINE_HEATER.search(line)
        if inline and current_section:
            full_item_name = f"{current_section} {inline.group('capacity')} {inline.group('kw').upper()}"
            results.append({
                "item": full_item_name.strip(),
                "hours": float(inline.group("hours")),
                "unit": inline.group("unit"),
                "material_cost": clean_money(inline.group("material")),
                "labor_cost": clean_money(inline.group("labor")),
                "equipment_cost": 0.0,
                "total_cost": clean_money(inline.group("total")),
            })
            continue

        # -----------------------------
        # SPEC_ROW (two-line heaters with spec)
        # -----------------------------
        spec = SPEC_ROW.search(line)
        if spec and current_section:
            parts = [current_section] + multi_line_buffer
            full_item_name = f"{' '.join(parts)} {spec.group('spec')}"
            multi_line_buffer = []

            results.append({
                "item": full_item_name.strip(),
                "hours": float(spec.group("hours")),
                "unit": spec.group("unit"),
                "material_cost": clean_money(spec.group("material")),
                "labor_cost": clean_money(spec.group("labor")),
                "equipment_cost": 0.0,
                "total_cost": clean_money(spec.group("total")),
            })
            continue

    return results

# ============================
# JSON Append Logic
# ============================

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
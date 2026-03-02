import json
import re

INPUT_FILE = "./data/painting_data.json"
OUTPUT_FILE = "./data/painting_finetune.jsonl"

SYSTEM_PROMPT = (
    "You are a professional construction Painting cost estimator. "
    "Location: United States. Currency: USD. "
    "Pricing basis year: 2024. "
    "Assume RSMeans-style unit pricing and productivity. "
    "Apply national average pricing (location factor = 1.00) "
    "Estimate labor hours, unit of measure, unit material cost, "
    "labor rate, and equipment rate based on standard U.S. construction practices. "
    "Return only structured JSON. Do not calculate totals. "
    "Do not include explanations."
)

def clean_material(material: str) -> str:
    """Remove (material #<number>) from material string."""
    if not material:
        return ""
    return re.sub(r"\s*\(material\s*#\w+\)", "", material, flags=re.IGNORECASE).strip()

def build_chat_entry(record):
    
    material_clean = clean_material(record.get("material", ""))
    item = f"{record.get('category', '')}, {record.get('application', '')}, {material_clean}, {record.get('speed', '')}"
    unit = record.get("unit", "SF")
    multiple = record.get("multiple", 100)
    if multiple is None:
        multiple = 100
    if unit not in ["SF", "LF"]:
        multiple = 1

    labor_units = record.get("labor_units_per_hour")
    labor_hours = round(1 / labor_units, 2) if labor_units else 0
    material_cost = record.get("material_cost_per_100unit", 0.0)
    labor_cost = record.get("labor_cost_per_100unit", 0.0)

    assistant_content = {
        "unit": record.get("unit"),
        "labor_hours": labor_hours,
        "unit_material_cost": round(record.get("material_cost_per_100unit", 0.0) / multiple, 2) if material_cost else 0.0,
        "unit_labor_rate": round(record.get("labor_cost_per_100unit", 0.0) / multiple, 2) if labor_cost else 0.0,
        "unit_equipment_rate": 0,
        "speed": record.get("speed"),
        "price_year": 2024,
        "Currency": "USD"
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": item
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_content)
            }
        ]
    }

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in data:
            entry = build_chat_entry(record)
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Created fine-tuning dataset: {OUTPUT_FILE}")
    print(f"📦 Total entries: {len(data)}")

if __name__ == "__main__":
    main()
import json

INPUT_FILE = "./data/hvac.json"
OUTPUT_FILE = "./data/hvac_finetune.jsonl"

SYSTEM_PROMPT = (
    "You are a professional construction Plumbing and HVAC cost estimator. "
    "Location: United States. Currency: USD. "
    "Pricing basis year: 2023. "
    "Assume RSMeans-style unit pricing and productivity. "
    "Apply national average pricing (location factor = 1.00) "
    "Estimate labor hours, unit of measure, unit material cost, "
    "labor rate, and equipment rate based on standard U.S. construction practices. "
    "Return only structured JSON. Do not calculate totals. "
    "Do not include explanations."
)

def build_chat_entry(record):

    assistant_content = {
        "unit": record.get("unit"),
        "labor_hours": record.get("labor_hours", 0),
        "unit_material_cost": record.get("material_cost", 0.0),
        "labor_rate": record.get("labor_cost", 0.0),
        "equipment_rate": record.get("equipment_cost", 0.0)
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": record.get("item")
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_content, indent=2)
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
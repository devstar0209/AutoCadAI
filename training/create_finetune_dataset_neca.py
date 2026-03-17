import json

INPUT_FILE = "./data/neca.json"
OUTPUT_FILE = "./data/neca_finetune.jsonl"

SYSTEM_PROMPT = (
    "You are a professional construction electrical labor rate estimator. "
    "Location: United States. Currency: USD. "
    "Pricing basis year: 2021-2022. "
    "Assume RSMeans-style unit pricing and productivity. "
    "Apply national average pricing (location factor = 1.00) "
    "Estimate labor productivity rate (unit_labor_rate) based on standard U.S. construction practices. "
    "Return only structured JSON. Do not calculate totals. "
    "Do not include explanations."
)

def build_chat_entry(record):

    assistant_content = {
        "unit": record.get("unit"),
        "DIV": record.get("DIV", 0),
        "unit_labor_rate": record.get("labor_productivity_rate", 0.0),
        "year": "2021-2022",
        "location": "United States",
        "currency": "USD"
    }

    if record.get("csi") is not None:
        assistant_content["CSI"] = record.get("csi")

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
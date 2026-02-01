import os
import json
import math
import mysql.connector
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -----------------------------------
# CONFIG
# -----------------------------------
BULK_SIZE = 50
OUTPUT_FILE = "resources_enriched.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


ALLOWED_CATEGORIES = [
    "General Requirements",
    "Existing Conditions",
    "Concrete",
    "Masonry",
    "Metal",
    "Wood, Plastics, and Composites",
    "Thermal and Moisture Protection",
    "Openings",
    "Finishes",
    "Specialties",
    "Equipment",
    "Furnishing",
    "Special Construction",
    "Conveying Systems",
    "Fire Suppression",
    "Plumbing",
    "HVAC",
    "Electrical",
    "Communications",
    "Electronic Safety & Security",
    "Earthwork",
    "Exterior Improvement",
    "Utilities",
    "Transportations",
    "Waterway & marine",
    "Material Processing & Handling Equipment",
    "Pollution Control Equipment"
]

# -----------------------------------
# DB CONNECTION
# -----------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
    )

# -----------------------------------
# FETCH DATA
# -----------------------------------
def fetch_resource_components():
    """
    resource_components:
      resource_type | unit | category | orignal_rate
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            resource_type,
            unit,
            category,
            orignal_rate
        FROM resource_components
        ORDER BY resource_type
    """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# -----------------------------------
# BUILD ITEMS (ONE PER resource_type)
# -----------------------------------
def build_base_items(rows):
    """
    One JSON item per resource_type
    Aggregate Material / Labor / Equipment
    """
    items = {}

    for resource_type, unit, category, rate in rows:
        if resource_type not in items:
            items[resource_type] = {
                "item": resource_type,
                "unit": unit,
                "material_unit_cost": 0.0,
                "labor_rate": 0.0,
                "equipment_rate": 0.0,
                "DIV": None,
                "CSI": None,
                "Category": None
            }

        value = float(rate or 0)
        cat = category.lower()

        if cat == "material":
            items[resource_type]["material_unit_cost"] += value
        elif cat in ("labor", "labour"):
            items[resource_type]["labor_rate"] += value
        elif cat == "equipment":
            items[resource_type]["equipment_rate"] += value

    return items

# -----------------------------------
# JSON LOAD / SAVE (INCREMENTAL)
# -----------------------------------
def load_existing_output(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {i["item"]: i for i in data}

def save_output(items, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(items.values()), f, indent=2)

# -----------------------------------
# OPENAI CLASSIFICATION
# -----------------------------------
def classify_resource_types(resource_types):
    system_prompt = f"""
You are a senior construction estimator and CSI MasterFormat expert.

For each construction item:
- Assign CSI Division (DIV)
- Assign CSI MasterFormat number (CSI)
- Assign a high-level construction Category

**Allow categories are:** {', '.join(ALLOWED_CATEGORIES)}

Rules:
- Use CSI MasterFormat 2020+
- Best industry match
- Concise
- Return ONLY valid JSON

Validate:
- Invalid if CSI, DIV, or Category is NULL or empty
- Category must be one of the allowed categories
"""

    user_prompt = json.dumps({"items": resource_types})

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "csi_classification",
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item": {
                                    "type": "string"
                                },
                                "DIV": {
                                    "type": "string"
                                },
                                "CSI": {
                                    "type": "string"
                                },
                                "Category": {
                                    "type": "string"
                                }
                            },
                            "required": ["item", "DIV", "CSI", "Category"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False
            }
        }
    },
    )

    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(content)

    return {
        i["item"]: {
            "DIV": i["DIV"],
            "CSI": i["CSI"],
            "Category": i["Category"]
        }
        for i in parsed["items"]
    }

# -----------------------------------
# BULK ENRICHMENT (PER-BULK SAVE)
# -----------------------------------
def enrich_with_csi(items, output_file):
    existing = load_existing_output(output_file)

    # Restore saved classifications
    for item_name, saved in existing.items():
        if item_name in items and saved.get("CSI"):
            items[item_name].update({
                "DIV": saved["DIV"],
                "CSI": saved["CSI"],
                "Category": saved["Category"]
            })

    # Only unclassified items
    pending = [k for k, v in items.items() if not v["CSI"]]

    if not pending:
        print("[SKIP] All resource_types already classified")
        return items

    batches = math.ceil(len(pending) / BULK_SIZE)

    for i in range(batches):
        batch = pending[i * BULK_SIZE:(i + 1) * BULK_SIZE]
        print(f"[GPT] Batch {i + 1}/{batches} ({len(batch)} resource_types)")

        result = classify_resource_types(batch)

        for item_name, meta in result.items():
            if item_name in items:
                items[item_name].update(meta)

        # 🔹 SAVE AFTER EACH BULK
        save_output(items, output_file)
        print(f"[SAVE] JSON updated after batch {i + 1}")

    return items

# -----------------------------------
# MAIN
# -----------------------------------
def main():
    print("[DB] Fetching rows...")
    rows = fetch_resource_components()

    print("[MAP] Aggregating categories per resource_type...")
    items = build_base_items(rows)

    print("[GPT] Classifying resource_types (incremental)...")
    enrich_with_csi(items, OUTPUT_FILE)

    print(f"[DONE] Output written to {OUTPUT_FILE}")

# -----------------------------------
if __name__ == "__main__":
    main()

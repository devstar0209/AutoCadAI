import json
import os
import time
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# ==========================
# CONFIG
# ==========================
INPUT_FILE = "resources_csi_1.json"

BATCH_SIZE = 50
MODEL = "gpt-5.2"

# COUNTRIES = [
#     "US", "Barbados", "Belize",
#     "Dominica", "Cayman Islands", "Jamaica",
#     "Trinidad and Tobago"
# ]

COUNTRIES = [
    "Jamaica"
]


CURRENCY_MAP = {
    "US": "USD",
    "CA": "CAD",
    "UK": "GBP",
    "IE": "EUR",
    "EU": "EUR",
    "Antigua & Barbuda": "ECD",
    "Barbados": "BBD",
    "Belize": "BZD",
    "Dominica": "ECD",
    "Cayman Islands": "KYD",
    "Dominica": "ECD",
    "Jamaica": "JMD",
    "St Lucia": "ECD",
    "Trinidad and Tobago": "TTD",
}

SLEEP_SECONDS = 2

# ==========================
# INIT
# ==========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================
# HELPERS
# ==========================
def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def chunks(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

def pricing_standard(country: str) -> str:
    return "CSI" if country in ["US", "CA"] else "NRM2"

# ==========================
# GPT CALL
# ==========================
def call_gpt(system_prompt: str, payload: dict) -> list:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)}
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
                                "material_unit_cost": {
                                    "type": "number"
                                },
                                "labor_rate": {
                                    "type": "number"
                                },
                                "equipment_rate": {
                                    "type": "number"
                                },
                                "qty": {
                                    "type": "number"
                                },
                                "unit": {
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
                            "required": ["item", "DIV", "CSI", "Category","qty","unit", "material_unit_cost", "labor_rate", "equipment_rate"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False
            }
        }
        }
    )

    try:
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)["items"]
    except Exception:
        raise RuntimeError("Model returned invalid JSON")

# ==========================
# CSI MODE
# ==========================
CSI_SYSTEM_PROMPT = """
You are a US construction cost estimator.

Rules:
- Follow CSI MasterFormat
- Assume RSMeans-style unit pricing
- Provide conceptual market estimates (not licensed data)
- Labor and equipment are installation costs
- Apply city location factor
- Output normalized pricing schema
- Do not explain, output JSON only

COST NORMALIZATION RULE (MANDATORY):
- material_unit_cost MUST be the cost per ONE unit as defined by `unit`
- NEVER output extended, lot, reel, coil, or total material cost
- If an item is commonly sold in rolls, reels, or lots (e.g., 500 ft):
    - Divide total package price by total length
    - Output cost per 1 LF (or per 1 EA if unit = EA)
- Total material cost must ALWAYS equal:
    material_unit_cost × qty
- If unsure, normalize to the smallest practical install unit (LF, SF, EA)
"""

def estimate_csi(batch, location):
    payload = {
        "standard": "CSI",
        "location": location,
        "currency": CURRENCY_MAP[location["country"]],
        "items": batch
    }
    return call_gpt(CSI_SYSTEM_PROMPT, payload)

# ==========================
# NRM2 MODE
# ==========================
NRM2_SYSTEM_PROMPT = """
You are a construction cost estimator.

Rules:
- Follow CSI MasterFormat
- Assume NRM2/RICS-style unit pricing
- Provide conceptual market estimates (not licensed data)
- Labor and equipment are installation costs
- Apply reginal location factor
- Output normalized pricing schema
- Do not explain, output JSON only

COST NORMALIZATION RULE (MANDATORY):
- material_unit_cost MUST be the cost per ONE unit as defined by `unit`
- NEVER output extended, lot, reel, coil, or total material cost
- If an item is commonly sold in rolls, reels, or lots (e.g., 500 ft):
    - Divide total package price by total length
    - Output cost per 1 LF (or per 1 EA if unit = EA)
- Total material cost must ALWAYS equal:
    material_unit_cost × qty
- If unsure, normalize to the smallest practical install unit (LF, SF, EA)
"""

def estimate_nrm2(batch, location):
    payload = {
        "standard": "NRM2",
        "location": location,
        "currency": CURRENCY_MAP[location["country"]],
        "items": batch
    }
    return call_gpt(NRM2_SYSTEM_PROMPT, payload)

# ==========================
# MAIN
# ==========================
def main():
    all_data = load_json(INPUT_FILE)

    for country in COUNTRIES:
        print(f"\n--- Processing country: {country} ---")

        location = {"country": country}
        currency = CURRENCY_MAP[country]

        # Load previous results or start empty
        output_file = f"resources_priced_{currency.lower()}.json"
        try:
            results = load_json(output_file)
        except FileNotFoundError:
            results = []

        standard = pricing_standard(country)
        print(f"Pricing mode: {standard}, currency: {currency}")

        # Process in batches
        for batch in chunks(all_data, BATCH_SIZE):
            # enriched = estimate_csi(batch, country)
            if standard == "CSI":
                enriched = estimate_csi(batch, location)
            else:
                enriched = estimate_nrm2(batch, location)

            results.extend(enriched)
            save_json(output_file, results)
            time.sleep(SLEEP_SECONDS)

        print(f"Completed pricing → {output_file}")

if __name__ == "__main__":
    main()

import re
import json
import os
from typing import List, Dict
from openpyxl import Workbook
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Make sure this is set
MAX_TOKENS = 1400
OVERLAP_TOKENS = 200
QUANTITY_PRIORITY = {"stated": 3, "derived": 2, "assumed": 1}
UNIT_ALIASES = {"sf":"SF","sqft":"SF","square feet":"SF","cy":"CY","lf":"LF","ea":"EA"}

# RSMeans example database
RS_MEANS = [
    {"code":"03 30 00.10","name":"Concrete Slab on Grade,4 in","unit":"SF","keywords":["slab","grade","concrete","4 in"]},
    {"code":"03 15 00.20","name":"Concrete Footings","unit":"CY","keywords":["footing","concrete","grade"]}
]

RS_MEANS_COSTS = {
    "03 30 00.10":{"unit_cost":7.5,"labor_cost":3.25,"unit":"SF"},
    "03 15 00.20":{"unit_cost":120.0,"labor_cost":50.0,"unit":"CY"}
}

SECTION_HEADERS = [
    r"GENERAL NOTES",
    r"SPECIAL NOTES",
    r"LEGEND",
    r"CONCRETE NOTES",
    r"MECHANICAL NOTES",
    r"ELECTRICAL NOTES",
    r"\bSECTION\s+\d{2}"
]

# ============================================================
# UTILITIES
# ============================================================
def normalize_ocr(text: str) -> str:
    text = re.sub(r"[•○*]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sections(text: str) -> List[Dict]:
    sections = []
    current = {"title": "UNCLASSIFIED", "text": ""}
    for line in text.splitlines():
        if any(re.search(h, line.upper()) for h in SECTION_HEADERS):
            sections.append(current)
            current = {"title": line.strip(), "text": ""}
        else:
            current["text"] += line + " "
    sections.append(current)
    print(f"[INFO] Split text into {len(sections)} sections")
    return sections

def is_note_section(title: str) -> bool:
    return "NOTE" in title.upper() or "LEGEND" in title.upper()

def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start
        token_count = 0
        while end < len(words) and token_count < MAX_TOKENS:
            token_count += 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        start = max(end - OVERLAP_TOKENS, end)
    print(f"[INFO] Chunked text into {len(chunks)} chunks")
    return chunks

def normalize_unit(u:str)->str: return UNIT_ALIASES.get(u.lower(), u.upper())

def parse_quantity(component:Dict)->float:
    units = normalize_unit(component.get("units","SF"))
    qty = 1.0
    dims = component.get("dimensions",{})

    if "area" in dims:
        m = re.search(r"([\d,.]+)", dims["area"])
        if m: qty = float(m.group(1).replace(",",""))

    if units=="SF" and "thickness" in dims:
        m = re.search(r"([\d,.]+)", dims["thickness"])
        if m:
            thickness_in = float(m.group(1))
            qty = qty * (thickness_in/12)/27  # SF * thickness_ft / 27 = CY
            units="CY"
            component["units"]="CY"
    print(f"[INFO] Parsed quantity {qty} {units} for component '{component['component_name']}'")
    return round(qty,3)

def system_fingerprint(system: Dict) -> str:
    mats = sorted(c.get("material","").lower() for c in system["components"])
    return f'{system["csi_division"]}|{system["system_name"].lower()}|{"-".join(mats)}'

def component_fingerprint(c: Dict) -> str:
    return f'{c["component_name"].lower()}|{c.get("material","").lower()}|{c.get("units","")}'

def merge_components(base: Dict, incoming: Dict) -> Dict:
    if QUANTITY_PRIORITY[incoming["quantity_basis"]] > QUANTITY_PRIORITY[base["quantity_basis"]]:
        base["dimensions"] = incoming.get("dimensions")
        base["quantity_basis"] = incoming["quantity_basis"]
    base["notes"] = list(set(base.get("notes", []) + incoming.get("notes", [])))
    base["occurrences"] += 1
    return base

def merge_systems(base: Dict, incoming: Dict) -> Dict:
    base["scope_included"] = list(set(base["scope_included"] + incoming["scope_included"]))
    base["scope_excluded"] = list(set(base["scope_excluded"] + incoming["scope_excluded"]))
    base["chunk_count"] += 1
    for k in ["codes","standards","tolerances"]:
        base["constraints"][k] = list(set(base["constraints"].get(k,[]) + incoming["constraints"].get(k,[])))
    comp_map = {component_fingerprint(c): c for c in base["components"]}
    for c in incoming["components"]:
        key = component_fingerprint(c)
        if key in comp_map:
            comp_map[key] = merge_components(comp_map[key], c)
        else:
            c["occurrences"] = 1
            comp_map[key] = c
    base["components"] = list(comp_map.values())
    return base

def component_confidence(c: Dict) -> float:
    qty = {"stated":1.0,"derived":0.7,"assumed":0.3}[c["quantity_basis"]]
    rep = min(1.0, c["occurrences"]/3)
    notes = 1.0 if c.get("notes") else 0.5
    return round(0.45*qty + 0.35*rep + 0.2*notes,2)

def system_confidence(s: Dict) -> float:
    comp_avg = sum(component_confidence(c) for c in s["components"])/len(s["components"])
    agreement = min(1.0, s["chunk_count"]/3)
    scope = 1.0 if len(s["scope_included"])>=2 else 0.6
    return round(0.5*comp_avg + 0.3*agreement + 0.2*scope,2)

def rsmeans_match(system:Dict)->Dict:
    text = (system["system_name"] + " " + " ".join(system["scope_included"])).lower()
    candidates = []
    for r in RS_MEANS:
        hits = sum(1 for k in r["keywords"] if k in text)
        score = hits/len(r["keywords"])
        if score>0:
            cost_data = RS_MEANS_COSTS.get(r["code"],{"unit_cost":0,"labor_cost":0,"unit":r["unit"]})
            candidates.append({
                "code": r["code"], "name": r["name"], "unit": cost_data["unit"],
                "fit_score": round(score,2), "unit_cost": cost_data["unit_cost"],
                "labor_cost": cost_data["labor_cost"]
            })
    if not candidates:
        print(f"[RSMeans] No match found for system '{system['system_name']}'")
        return {"code":None,"name":None,"unit":None,"fit_score":0,"unit_cost":0,"labor_cost":0}
    best = max(candidates, key=lambda x:x["fit_score"])
    print(f"[RSMeans] Matched system '{system['system_name']}' -> '{best['name']}' (fit {best['fit_score']})")
    return best

def calculate_cost(system:Dict):
    total_mat,total_lab = 0.0,0.0
    for c in system["components"]:
        qty = parse_quantity(c)
        uc = system["rsmeans"]["unit_cost"]
        lc = system["rsmeans"]["labor_cost"]
        total_mat += uc*qty
        total_lab += lc*qty
        c["quantity"] = qty
        c["material_cost"] = round(uc*qty,2)
        c["labor_cost"] = round(lc*qty,2)
        c["total_cost"] = round(uc*qty + lc*qty,2)
        print(f"[COST] Component '{c['component_name']}': Material ${c['material_cost']}, Labor ${c['labor_cost']}, Total ${c['total_cost']}")
    system["material_cost"] = round(total_mat,2)
    system["labor_cost"] = round(total_lab,2)
    system["total_cost"] = round(total_mat + total_lab,2)
    print(f"[COST] System '{system['system_name']}': Total Material ${system['material_cost']}, Labor ${system['labor_cost']}, Total ${system['total_cost']}")
    return system

def export_excel(systems:List[Dict],filename:str):
    wb = Workbook()
    ws = wb.active
    ws.title = "System Takeoff"
    ws.append(["System","RSMeans Code","RSMeans Name","Fit","System Confidence",
               "Component","Material","Unit","Quantity","Quantity Basis","Component Confidence",
               "Material Cost","Labor Cost","Total Cost","Notes"])
    for s in systems:
        for c in s["components"]:
            ws.append([s["system_name"],s["rsmeans"]["code"],s["rsmeans"]["name"],s["rsmeans"]["fit_score"],
                       s["confidence"],c["component_name"],c.get("material"),c["units"],
                       c.get("quantity",1),c["quantity_basis"],c["confidence"],
                       c["material_cost"],c["labor_cost"],c["total_cost"],"; ".join(c.get("notes",[]))])
    wb.save(filename)
    print(f"[INFO] Exported Excel to '{filename}'")

# ============================================================
# GPT RESPONSES PIPELINE
# ============================================================
def call_gpt_responses_classify(text: str) -> List[Dict]:
    print("[GPT] Classifying system assemblies via Chat Completions API...")

    system_prompt = """
    You are an expert construction estimator.
    Identify system assemblies from construction text.
    Only one assembly per distinct system-location combination.
    Return ONLY valid JSON matching the schema.
    """

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0,
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": text }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "system_assemblies",
                "schema": {
                    "type": "object",
                    "properties": {
                        "assemblies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "assembly_name": {"type": "string"},
                                    "csi_division": {"type": "string"},
                                    "confidence_hint": {"type": "number"}
                                },
                                "required": ["assembly_name", "csi_division"]
                            }
                        }
                    },
                    "required": ["assemblies"]
                }
            }
        }
    )

    data = response.choices[0].message.content

    data = json.loads(data)
    print(f"[GPT] Found systems", data)
    if not isinstance(data, dict) or "assemblies" not in data:
        raise ValueError("Invalid structured output from GPT")

    return data["assemblies"]


def call_gpt_responses_extract(text: str, assembly_name: str) -> Dict:
    print(f"[GPT] Extracting system '{assembly_name}' details...")

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior construction estimator.\n"
                    "Rules:\n"
                    "- Extract ONLY information supported by the text\n"
                    "- Do NOT invent quantities or dimensions\n"
                    "- Return ONLY valid JSON matching the schema\n"
                )
            },
            {
                "role": "user",
                "content": f"SYSTEM ASSEMBLY: {assembly_name}\n\n{text}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "system_assembly",
                "schema": {
                    "type": "object",
                    "properties": {
                        "system_name": {"type": "string"},
                        "csi_division": {"type": "string"},
                        "components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "component_name": {"type": "string"},
                                    "material": {"type": "string"},
                                    "units": {"type": "string"},
                                    "quantity_basis": {
                                        "type": "string",
                                        "enum": ["stated", "derived", "assumed"]
                                    },
                                    "dimensions": {
                                        "type": "object",
                                        "additionalProperties": {"type": "string"}
                                    },
                                    "notes": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": [
                                    "component_name",
                                    "units",
                                    "quantity_basis"
                                ]
                            }
                        },
                        "scope_included": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "scope_excluded": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "codes": {"type": "array", "items": {"type": "string"}},
                                "standards": {"type": "array", "items": {"type": "string"}},
                                "tolerances": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "required": ["system_name", "components"]
                }
            }
        }
    )

    data = json.loads(response.choices[0].message.content)

    if not isinstance(data, dict):
        raise ValueError("Invalid structured output from GPT")

    print(f"[GPT] Extracted {len(data['components'])} components")
    return data

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(text:str)->List[Dict]:
    print("[PIPELINE] Starting pipeline...")
    text = normalize_ocr(text)
    sections = split_sections(text)
    notes_text = " ".join(s["text"] for s in sections if is_note_section(s["title"]))
    raw=[]
    for s in sections:
        if is_note_section(s["title"]): continue
        for chunk in chunk_text(s["text"]):
            llm_input = f"GENERAL NOTES:\n{notes_text}\nSYSTEM TEXT:\n{chunk}"
            assemblies = call_gpt_responses_classify(llm_input)
            for a in assemblies:
                sys = call_gpt_responses_extract(llm_input, a["assembly_name"])
                sys["chunk_count"] = 1
                for c in sys["components"]:
                    c["occurrences"] = 1
                    c["units"] = normalize_unit(c["units"])
                raw.append(sys)
    print(f"[PIPELINE] Extracted {len(raw)} raw systems")
    merged={}
    for s in raw:
        k = system_fingerprint(s)
        merged[k] = s if k not in merged else merge_systems(merged[k], s)
    print(f"[PIPELINE] Merged into {len(merged)} unique systems")
    final=[]
    for s in merged.values():
        for c in s["components"]:
            c["confidence"] = component_confidence(c)
        s["confidence"] = system_confidence(s)
        s["rsmeans"] = rsmeans_match(s)
        s = calculate_cost(s)
        final.append(s)
    print("[PIPELINE] Pipeline complete")
    return final

# ============================================================
# MAIN
# ============================================================
if __name__=="__main__":
    with open("input.txt","r",encoding="utf-8") as f: text=f.read()
    systems = run_pipeline(text)
    with open("system_takeoff.json","w",encoding="utf-8") as f: json.dump(systems,f,indent=2)
    export_excel(systems,"system_takeoff.xlsx")
    print("✅ Production takeoff pipeline complete.")

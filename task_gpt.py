import json
import re
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gpt-5.2"
TEMPERATURE = 0

MAX_CHARS = 6000       # ~1500 tokens
OVERLAP_CHARS = 800    # ~200 tokens

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# STEP 0 — CONSTRUCTION-AWARE CHUNKING
# ============================================================

def chunk_construction_text(text: str) -> List[str]:
    """
    Chunk construction text using paragraph-aware overlapping windows.
    """

    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + MAX_CHARS, length)

        split = text.rfind("\n\n", start, end)
        if split == -1 or split <= start:
            split = end

        chunk = text[start:split].strip()
        if chunk:
            chunks.append(chunk)

        start = split - OVERLAP_CHARS
        if start < 0:
            start = 0

    return chunks


# ============================================================
# STEP 1 — SYSTEM CLASSIFICATION (MACRO)
# ============================================================

def classify_systems(chunk: str) -> List[Dict]:
    """
    Identify macro construction systems from a text chunk.
    """

    system_prompt = """
    You are a senior construction estimator.

    OBJECTIVE:
    Identify ONLY high-level construction SYSTEMS (macro scale).

    STRICT RULES:
    - Output macro systems only (e.g., Stair System, Electrical System).
    - Do NOT output components, materials, or minor assemblies.
    - Do NOT output quantities, units, or costs.
    - One record per system.
    - Normalize to standard industry terminology.
    - Prefer CSI MasterFormat naming.

    ALLOWED:
    Structural Framing System, Stair System, Roofing System,
    Electrical System, Plumbing System, HVAC System,
    Fire Protection System, Exterior Wall System,
    Interior Partition System

    DISALLOWED:
    Steel stringers, conduit, rebar, handrails, fixtures

    Return ONLY valid JSON matching the schema.
    """

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "construction_systems",
                "schema": {
                    "type": "object",
                    "properties": {
                        "systems": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "system_name": {"type": "string"},
                                    "csi_division": {"type": "string"},
                                    "included_categories": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": [
                                    "system_name",
                                    "csi_division",
                                    "included_categories"
                                ]
                            }
                        }
                    },
                    "required": ["systems"]
                }
            }
        }
    )

    return json.loads(response.choices[0].message.content)["systems"]


def classify_systems_chunked(text: str) -> List[Dict]:
    """
    Run system classification across all chunks and deduplicate globally.
    """

    chunks = chunk_construction_text(text)
    all_systems = []

    for chunk in chunks:
        all_systems.extend(classify_systems(chunk))

    unique = {}
    for sys in all_systems:
        key = sys["system_name"].lower()
        unique[key] = sys

    return list(unique.values())


# ============================================================
# STEP 2 — SYSTEM → ASSEMBLIES (MESO)
# ============================================================

def relevant_chunks_for_system(system_name: str, chunks: List[str]) -> str:
    """
    Filter chunks relevant to a given system.
    """

    keywords = system_name.lower().split()
    return "\n\n".join(
        c for c in chunks if any(k in c.lower() for k in keywords)
    )


def map_system_to_assemblies(system: Dict, full_text: str) -> List[Dict]:
    """
    Decompose a system into estimatable assemblies.
    """

    system_prompt = """
    You are a chief construction estimator with RSMeans expertise.

    OBJECTIVE:
    Decompose a construction SYSTEM into standard ESTIMATING ASSEMBLIES.

    RULES:
    - Assemblies must be estimatable work packages.
    - Assemblies map cleanly to RSMeans or NECA sections.
    - Do NOT include materials, fasteners, or line items.
    - Do NOT include quantities, units, or costs.
    - Use standard CSI MasterFormat divisions.
    - Avoid overlaps and duplicates.

    Return ONLY valid JSON matching the schema.
    """

    chunks = chunk_construction_text(full_text)
    context = relevant_chunks_for_system(system["system_name"], chunks)

    if not context:
        context = full_text[:MAX_CHARS]

    user_prompt = f"""
    SYSTEM NAME:
    {system["system_name"]}

    CSI DIVISION:
    {system["csi_division"]}

    INCLUDED CATEGORIES:
    {", ".join(system["included_categories"])}

    SOURCE CONTEXT:
    {context}
    """

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
                                    "description": {"type": "string"}
                                },
                                "required": [
                                    "assembly_name",
                                    "csi_division"
                                ]
                            }
                        }
                    },
                    "required": ["assemblies"]
                }
            }
        }
    )

    assemblies = json.loads(response.choices[0].message.content)["assemblies"]

    # Dedup assemblies
    seen = set()
    unique = []
    for a in assemblies:
        key = (a["assembly_name"].lower(), a["csi_division"])
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique


# ============================================================
# STEP 3 — ASSEMBLY → LINE ITEMS & UNITS (MICRO)
# ============================================================

def map_assembly_to_line_items(assembly: Dict) -> List[Dict]:
    """
    Convert an assembly into measurable line items.
    """

    system_prompt = """
    You are an expert construction takeoff technician.

    OBJECTIVE:
    Convert a construction ASSEMBLY into measurable LINE ITEMS.

    RULES:
    - Use industry-standard measurement units only.
    - Do NOT include prices, rates, or quantities.
    - Items must be measurable and database-ready.
    - Use RSMeans / NECA compatible terminology.

    STANDARD UNITS:
    LF, SF, SY, CY, EA, LB, TON, HR

    Return ONLY valid JSON matching the schema.
    """

    user_prompt = f"""
    ASSEMBLY NAME:
    {assembly["assembly_name"]}

    CSI DIVISION:
    {assembly["csi_division"]}
    """

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "assembly_line_items",
                "schema": {
                    "type": "object",
                    "properties": {
                        "line_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item_name": {"type": "string"},
                                    "unit": {"type": "string"},
                                    "labor_unit": {"type": "string"},
                                    "material_unit": {"type": "string"}
                                },
                                "required": ["item_name", "unit"]
                            }
                        }
                    },
                    "required": ["line_items"]
                }
            }
        }
    )

    return json.loads(response.choices[0].message.content)["line_items"]


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

def run_pipeline(source_text: str) -> Dict:
    """
    Full macro → meso → micro pipeline with chunking.
    """

    output = {"systems": []}

    systems = classify_systems_chunked(source_text)
    print("Systems::", systems)

    for system in systems:
        assemblies = map_system_to_assemblies(system, source_text)

        for assembly in assemblies:
            assembly["line_items"] = map_assembly_to_line_items(assembly)

        system["assemblies"] = assemblies
        output["systems"].append(system)

    return output


# ============================================================
# EXAMPLE EXECUTION
# ============================================================

if __name__ == "__main__":
    with open("input.txt","r",encoding="utf-8") as f: text=f.read()

    result = run_pipeline(text)
    with open("result.json","w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print("Done successfully!!!")

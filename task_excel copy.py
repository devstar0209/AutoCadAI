import os
import re
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Optional but recommended for approximate token sizing
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENC = None

from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------

ALLOWED_DIVISIONS = {
    "01": "General Requirements",
    "02": "Existing Conditions",
    "03": "Concrete",
    "04": "Masonry",
    "05": "Metals",
    "06": "Wood, Plastics, and Composites",
    "07": "Thermal and Moisture Protection",
    "08": "Openings",
    "09": "Finishes",
    "10": "Specialties",
    "11": "Equipment",
    "12": "Furnishings",
    "13": "Special Construction",
    "14": "Conveying Equipment",
    "21": "Fire Suppression",
    "22": "Plumbing",
    "23": "HVAC",
    "25": "Integrated Automation",
    "26": "Electrical",
    "27": "Communications",
    "28": "Electronic Safety and Security",
    "31": "Earthwork",
    "32": "Exterior Improvements",
    "33": "Utilities",
}

SYSTEM_KEYWORDS = {
    "HVAC": ["hvac", "mechanical", "air handling", "ahu", "duct", "vav", "rtu", "exhaust", "supply air"],
    "Plumbing": ["plumbing", "domestic water", "sanitary", "storm", "water heater", "fixture", "grease"],
    "Fire Protection": ["fire protection", "fire suppression", "sprinkler", "standpipe", "nfpa"],
    "Electrical": ["electrical", "panel", "switchgear", "conduit", "wire", "receptacle", "lighting", "branch circuit"],
    "Low Voltage": ["low voltage", "data", "telecom", "communications", "cctv", "access control", "fire alarm"],
    "Architectural": ["architectural", "finishes", "gyp", "drywall", "tile", "paint", "doors", "windows"],
    "Structural": ["structural", "steel", "beam", "column", "rebar", "foundation", "concrete"],
    "Civil/Site": ["civil", "site", "grading", "paving", "curb", "drainage", "utility", "manhole"],
}

REGION_STANDARDS = {
    "US": "US",
    "USA": "US",
    "United States": "US",
    "Caribbean": "NRM2",
    "Commonwealth": "NRM2",
    "UK": "NRM2",
    "NRM2": "NRM2",
}

UNIT_MAP_US = {
    "sq ft": "SF", "sf": "SF", "ft2": "SF", "sqf": "SF",
    "lin ft": "LF", "lf": "LF", "ft": "LF",
    "cy": "CY", "yd3": "CY",
    "ea": "EA", "each": "EA", "nr": "EA",
    "ton": "TON",
}

UNIT_MAP_NRM2 = {
    "m2": "m²", "sqm": "m²", "sq m": "m²",
    "m": "m", "lm": "m",
    "m3": "m³",
    "nr": "nr", "no.": "nr", "each": "nr",
    "kg": "kg", "t": "t",
}

DEFAULTS = {
    "wall_height_us_ft": 10.0,
    "wall_height_nrm2_m": 3.0,
    "door_us_default": {"width_ft": 3.0, "height_ft": 7.0, "unit": "EA"},
    "door_nrm2_default": {"width_mm": 900, "height_mm": 2100, "unit": "nr"},
    "slab_sog_us_thickness_in": 4.0,
    "slab_sog_nrm2_thickness_mm": 100,
}

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")  # adjust to your deployed name

# -----------------------------
# Utilities
# -----------------------------

def approx_tokens(text: str) -> int:
    if _ENC:
        return len(_ENC.encode(text))
    # rough fallback: ~4 chars per token in English
    return max(1, len(text) // 4)

def normalize_region(region: str) -> str:
    return REGION_STANDARDS.get(region.strip(), "US")

def normalize_unit(unit: str, region_std: str) -> str:
    if not unit:
        return unit
    u = unit.strip().lower()
    if region_std == "US":
        return UNIT_MAP_US.get(u, unit)
    return UNIT_MAP_NRM2.get(u, unit)

def detect_systems_in_text(text: str) -> List[str]:
    t = text.lower()
    hits = []
    for sys_name, kws in SYSTEM_KEYWORDS.items():
        if any(kw in t for kw in kws):
            hits.append(sys_name)
    return hits

def parse_user_system_request(prompt: str) -> List[str]:
    """
    Returns list of requested system names if prompt is valid; else empty (meaning process all).
    Valid if it maps to at least one known system or includes Div XX.
    """
    if not prompt or not prompt.strip():
        return []
    p = prompt.strip().lower()

    requested = set()

    print("[Debug] Parsing user system request from prompt:", prompt)

    # Div-based
    div_match = re.findall(r"\bdiv(?:ision)?\s*([0-3]\d)\b", p)
    for d in div_match:
        if d == "21": requested.add("Fire Protection")
        if d == "22": requested.add("Plumbing")
        if d == "23": requested.add("HVAC")
        if d == "26": requested.add("Electrical")
        if d in ("27", "28"): requested.add("Low Voltage")
        if d in ("31", "32", "33"): requested.add("Civil/Site")
        if d in ("03", "05"): requested.add("Structural")
        if d in ("07", "08", "09"): requested.add("Architectural")

    # Keyword-based
    for sys_name, kws in SYSTEM_KEYWORDS.items():
        if any(kw in p for kw in kws) or sys_name.lower() in p:
            requested.add(sys_name)

    return sorted(requested)

# -----------------------------
# Chunking
# -----------------------------

SECTION_BOUNDARY_RE = re.compile(
    r"(?im)^\s*(GENERAL NOTES|SPECIAL NOTES|ABBREVIATIONS|LEGEND|CODE NOTES)\s*$|"
    r"^\s*(DIVISION\s+\d{1,2}.*)\s*$|"
    r"^\s*(SECTION\s+\d{2}\s+\d{2}\s+\d{2}.*)\s*$|"
    r"^\s*([A-Z]{1,2}-\d{3}.*)\s*$|"
    r"^\s*(MECHANICAL|HVAC|PLUMBING|ELECTRICAL|FIRE PROTECTION|STRUCTURAL|ARCHITECTURAL|CIVIL|SITE)\s*.*$"
)

def split_into_raw_sections(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (header, body) tuples.
    """
    lines = text.splitlines()
    indices = []
    for i, line in enumerate(lines):
        if SECTION_BOUNDARY_RE.search(line):
            indices.append(i)

    if not indices:
        return [("DOCUMENT", text)]

    indices = sorted(set([0] + indices + [len(lines)]))
    sections = []
    for a, b in zip(indices[:-1], indices[1:]):
        chunk_lines = lines[a:b]
        header = chunk_lines[0].strip() if chunk_lines else "SECTION"
        body = "\n".join(chunk_lines).strip()
        if body:
            sections.append((header, body))
    return sections

def extract_global_notes(sections: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    notes = []
    for header, body in sections:
        h = header.lower()
        if "general notes" in h:
            notes.append({"note_type": "general", "text": body})
        elif "special notes" in h:
            notes.append({"note_type": "special", "text": body})
        elif "legend" in h or "abbreviations" in h:
            notes.append({"note_type": "legend", "text": body})
        elif "code" in h:
            notes.append({"note_type": "code", "text": body})
    return notes

def size_control_chunks(sections: List[Tuple[str, str]],
                        target_tokens: int = 3500,
                        max_tokens: int = 6000,
                        overlap_chars: int = 500) -> List[Dict[str, Any]]:
    chunks = []
    buf = ""
    buf_headers = []

    def flush():
        nonlocal buf, buf_headers
        if buf.strip():
            chunks.append({
                "headers": buf_headers[:],
                "text": buf.strip()
            })
        buf = ""
        buf_headers = []

    for header, body in sections:
        section_text = f"{header}\n{body}\n"
        if approx_tokens(buf + section_text) > max_tokens and buf:
            flush()

        if not buf:
            buf_headers = [header]
            buf = section_text
        else:
            buf_headers.append(header)
            buf += section_text

        if approx_tokens(buf) >= target_tokens:
            flush()

    flush()

    # add overlap
    final = []
    prev_tail = ""
    for i, ch in enumerate(chunks, start=1):
        text = ch["text"]
        if prev_tail:
            text = prev_tail + "\n" + text
        prev_tail = ch["text"][-overlap_chars:]
        final.append({
            "chunk_id": f"chunk_{i:03d}",
            "headers": ch["headers"],
            "text": text
        })
    return final

# -----------------------------
# GPT extraction
# -----------------------------

EXTRACTION_INSTRUCTIONS = """
You extract construction takeoff line-items as SYSTEM ASSEMBLIES and map them to CSI MasterFormat divisions/categories.

Hard constraints:
- Output MUST be valid JSON matching the requested schema.
- Only use allowed divisions provided.
- Prefer explicit quantities; if missing, derive using defaults described (and mark quantity_basis="derived" and add assumptions).
- If cannot derive safely, set quantity=null and needs_review=true.
- Keep line items granular enough for assemblies (not too atomic like "screw", not too vague like "misc").
- Include masterformat_section if present; else best-effort.
- Region standard affects units (US=RSMeans style units, NRM2=metric units). Normalize units.
"""

def build_schema_prompt(allowed_divs: Dict[str, str]) -> str:
    allowed = [{"div": k, "category": v} for k, v in allowed_divs.items()]
    return f"""
Return JSON with keys:
- document_meta: {{project_name, region_standard, source_id}}
- global_notes: array of {{note_type, text}} (brief; do not paste everything)
- systems: array of {{
    system_name, system_code, chunks_used,
    line_items: array of {{
        div, category, masterformat_section, item,
        quantity, unit, quantity_basis, region_unit_standard,
        assumptions: array of {{assumption_id, text}},
        needs_review,
        source: {{chunk_id, excerpt}}
    }},
    system_notes: array of {{text}}
}}
- extraction_warnings: array of {{chunk_id, warning}}

Allowed divisions list:
{json.dumps(allowed, indent=2)}
""".strip()

def call_gpt_extract(client: OpenAI,
                     chunk: Dict[str, Any],
                     global_notes: List[Dict[str, str]],
                     region_std: str,
                     source_id: str) -> Dict[str, Any]:

    schema_prompt = build_schema_prompt(ALLOWED_DIVISIONS)

    user_payload = {
        "document_meta": {"project_name": "", "region_standard": region_std, "source_id": source_id},
        "global_notes": global_notes[:],
        "chunk": {"chunk_id": chunk["chunk_id"], "headers": chunk["headers"], "text": chunk["text"]},
        "defaults": DEFAULTS,
        "unit_guidance": {
            "region_standard": region_std,
            "examples_us": ["SF", "LF", "CY", "EA"],
            "examples_nrm2": ["m²", "m", "m³", "nr"]
        }
    }

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": EXTRACTION_INSTRUCTIONS},
            {"role": "user", "content": schema_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        # If your endpoint supports structured outputs, you can enforce JSON stricter here.
        # response_format={"type":"json_object"}
    )

    text = resp.output_text
    return json.loads(text)

# -----------------------------
# Post-processing
# -----------------------------

def coerce_allowed_div(div: str) -> Optional[str]:
    if not div:
        return None
    d = div.strip()
    if len(d) == 1:
        d = "0" + d
    if d in ALLOWED_DIVISIONS:
        return d
    # sometimes model returns "Div 23"
    m = re.search(r"(\d{2})", d)
    if m and m.group(1) in ALLOWED_DIVISIONS:
        return m.group(1)
    return None

def normalize_extraction(data: Dict[str, Any], region_std: str) -> Dict[str, Any]:
    for sys in data.get("systems", []):
        for li in sys.get("line_items", []):
            li["div"] = coerce_allowed_div(li.get("div")) or li.get("div")
            li["category"] = li.get("category") or (ALLOWED_DIVISIONS.get(li["div"], "") if li.get("div") else "")
            li["region_unit_standard"] = region_std
            li["unit"] = normalize_unit(li.get("unit", ""), region_std)
    return data

def to_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for sys in data.get("systems", []):
        sysname = sys.get("system_name", "UNSPECIFIED SYSTEM").strip() or "UNSPECIFIED SYSTEM"
        for li in sys.get("line_items", []):
            rows.append({
                "System": sysname,
                "DIV": li.get("div", ""),
                "Category": li.get("category", ""),
                "Item": li.get("item", ""),
                "Quantity": li.get("quantity", None),
                "Unit": li.get("unit", ""),
            })
    return rows

# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(
    input_txt_path: str,
    output_xlsx_path: str,
    region: str,
    source_id: str = "text_dump_001",
    user_custom_prompt: str = ""
):
    client = OpenAI()

    region_std = normalize_region(region)

    print("[Step 1] Read input text")
    with open(input_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print("[Step 2] Split into raw sections")
    sections = split_into_raw_sections(text)
    print(f"  Raw sections: {len(sections)}")

    print("[Step 3] Extract global notes")
    global_notes = extract_global_notes(sections)
    print(f"  Global note blocks: {len(global_notes)}")

    print("[Step 4] Size-control chunking")
    chunks = size_control_chunks(sections, target_tokens=1500, max_tokens=2000, overlap_chars=200)
    print(f"  Final chunks: {len(chunks)}")

    requested_systems = parse_user_system_request(user_custom_prompt)
    if requested_systems:
        print("[Step 5] User custom prompt valid -> filter chunks by requested systems")
        print(f"  Requested systems: {requested_systems}")
        filtered = []
        for ch in chunks:
            sys_hits = detect_systems_in_text(ch["text"])
            if any(s in sys_hits for s in requested_systems):
                filtered.append(ch)
        # fallback: if filtering removes everything, process all (safer)
        if filtered:
            chunks_to_process = filtered
            print(f"  Chunks selected: {len(chunks_to_process)} / {len(chunks)}")
        else:
            chunks_to_process = chunks
            print("  No chunks matched requested systems; processing all chunks instead.")
    else:
        print("[Step 5] No valid system request -> process all chunks")
        chunks_to_process = chunks

    all_rows = []
    all_warnings = []

    print("[Step 6] Extract per chunk with GPT")
    for i, ch in enumerate(chunks_to_process, start=1):
        print(f"  Processing {ch['chunk_id']} ({i}/{len(chunks_to_process)}) headers={ch['headers'][:2]}...")

        extracted = call_gpt_extract(
            client=client,
            chunk=ch,
            global_notes=global_notes,
            region_std=region_std,
            source_id=source_id
        )
        extracted = normalize_extraction(extracted, region_std)

        rows = to_rows(extracted)
        all_rows.extend(rows)

        for w in extracted.get("extraction_warnings", []):
            all_warnings.append(w)

        print(f"    Line items extracted: {len(rows)}")
        if all_warnings:
            print(f"    Total warnings so far: {len(all_warnings)}")

    print("[Step 7] Write Excel grouped by system")
    if not all_rows:
        print("  No line items extracted. Writing empty workbook.")
        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(columns=["DIV", "Category", "Item", "Quantity", "Unit"]).to_excel(writer, index=False, sheet_name="EMPTY")
        return

    df = pd.DataFrame(all_rows)

    # Clean system names for Excel sheet limitations
    def safe_sheet(name: str) -> str:
        name = re.sub(r"[\[\]\:\*\?\/\\]", "-", name)
        return name[:31] or "SYSTEM"

    with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
        for sysname, g in df.groupby("System"):
            out = g[["DIV", "Category", "Item", "Quantity", "Unit"]].copy()
            out.to_excel(writer, index=False, sheet_name=safe_sheet(sysname))

        # Optional warnings sheet
        if all_warnings:
            pd.DataFrame(all_warnings).to_excel(writer, index=False, sheet_name="WARNINGS")

    print(f"  Wrote: {output_xlsx_path}")
    if all_warnings:
        print("[Step 8] Warnings summary")
        for w in all_warnings[:20]:
            print(f"  - {w.get('chunk_id')}: {w.get('warning')}")
        if len(all_warnings) > 20:
            print(f"  ... {len(all_warnings) - 20} more")

# -----------------------------
# CLI usage example
# -----------------------------
if __name__ == "__main__":
    # Example:
    # export OPENAI_API_KEY="..."
    # python takeoff_pipeline.py
    INPUT_TXT = "normalize_text.txt"
    OUTPUT_XLSX = "takeoff_by_system.xlsx"
    REGION = "US"  # or "Caribbean" / "Commonwealth"
    USER_PROMPT = ""  # e.g. "Process only HVAC and Plumbing"

    run_pipeline(
        input_txt_path=INPUT_TXT,
        output_xlsx_path=OUTPUT_XLSX,
        region=REGION,
        source_id="job_001",
        user_custom_prompt=USER_PROMPT
    )
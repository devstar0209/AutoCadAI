import os
import re
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz
import tiktoken

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------

MODEL = "gpt-5.2"  # adjust to your actual deployed name if different
MAX_OUTPUT_TOKENS = 3000

TARGET_CHUNK_TOKENS = 1600
CHUNK_OVERLAP_LINES = 12

# If user filter is valid, only process chunks with score >= this
FILTER_SCORE_THRESHOLD = 20

# Allowed categories (EDIT this to match your internal allowed list)
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

# Region mapping
def measurement_standard_for_region(region: str) -> str:
    r = (region or "").strip().lower()
    if r in ["us", "u.s.", "usa", "united states", "unitedstates"]:
        return "RSMeans"
    if r in ["caribbean", "jamaica", "bahamas", "barbados", "trinidad", "tobago",
             "cayman", "bermuda", "aruba", "curacao", "st lucia", "st. lucia",
             "grenada", "antigua", "dominica", "commonwealth", "uk", "u.k.", "united kingdom",
             "canada", "australia", "new zealand"]:
        return "NRM2"
    # default conservative
    return "NRM2"


# -----------------------------
# Chunking helpers
# -----------------------------

def get_encoder():
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return tiktoken.encoding_for_model("gpt-4o")  # fallback


ENC = get_encoder()

def estimate_tokens(text: str) -> int:
    return len(ENC.encode(text))


HEADER_PATTERNS = [
    re.compile(r"^\s*GENERAL\s+NOTES\b", re.I),
    re.compile(r"^\s*SPECIAL\s+NOTES\b", re.I),
    re.compile(r"^\s*KEY\s*NOTES\b", re.I),
    re.compile(r"^\s*LEGEND\b", re.I),
    re.compile(r"^\s*ABBREVIATIONS\b", re.I),
    re.compile(r"^\s*DIV(ISION)?\s*\d+\b", re.I),
    re.compile(r"^\s*\d{2}\s+\d{2}\s+\d{2}\b"),  # MasterFormat section like 07 21 00
    re.compile(r"^\s*[A-Z]{1,3}\d{1,3}(\.\d+)?\b"),  # sheet-ish like A101
]

def is_header_line(line: str) -> bool:
    return any(p.search(line) for p in HEADER_PATTERNS)

def split_into_blocks(text: str) -> List[str]:
    lines = text.splitlines()
    blocks = []
    buf = []
    for line in lines:
        if is_header_line(line) and buf:
            blocks.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b.strip()]

@dataclass
class Chunk:
    chunk_id: str
    text: str
    start_char: int
    end_char: int

def make_chunks(text: str, target_tokens: int = TARGET_CHUNK_TOKENS, overlap_lines: int = CHUNK_OVERLAP_LINES) -> List[Chunk]:
    blocks = split_into_blocks(text)

    chunks: List[Chunk] = []
    cur = ""
    cur_start = 0
    char_pos = 0
    chunk_index = 1

    prev_tail_lines: List[str] = []

    for block in blocks:
        block_with_sep = (block + "\n\n")
        block_tokens = estimate_tokens(block_with_sep)

        if not cur:
            cur_start = char_pos

        # If adding this block exceeds target and we already have content, flush
        if cur and (estimate_tokens(cur) + block_tokens > target_tokens):
            # finalize chunk with overlap already included
            chunk_text = cur.strip()
            end_char = cur_start + len(cur)
            chunks.append(Chunk(f"C{chunk_index:04d}", chunk_text, cur_start, end_char))
            chunk_index += 1

            # prepare overlap
            lines = chunk_text.splitlines()
            prev_tail_lines = lines[-overlap_lines:] if len(lines) > overlap_lines else lines[:]

            # start new chunk with overlap
            cur = "\n".join(prev_tail_lines).strip() + "\n"
            cur_start = end_char - len("\n".join(prev_tail_lines))  # approximate
        cur += block_with_sep
        char_pos += len(block_with_sep)

    if cur.strip():
        chunk_text = cur.strip()
        end_char = cur_start + len(cur)
        chunks.append(Chunk(f"C{chunk_index:04d}", chunk_text, cur_start, end_char))

    return chunks


# -----------------------------
# Filter validation and scoring
# -----------------------------

GENERIC_FILTERS = set([
    "all", "everything", "entire project", "entire", "whole project", "do it", "any", "none"
])

def is_valid_filter(filter_sentence: str) -> bool:
    if not filter_sentence:
        return False
    s = filter_sentence.strip().lower()
    if len(s.split()) < 4:
        return False
    if s in GENERIC_FILTERS:
        return False
    # must contain at least one alphanumeric keyword beyond stop-words
    keywords = [w for w in re.findall(r"[a-z0-9]+", s) if w not in {"the","and","or","for","with","only","just","all","from","into","to","of","in","on"}]
    return len(keywords) >= 2

def filter_score(filter_sentence: str, chunk_text: str) -> int:
    # simple scoring: fuzzy ratio + keyword hits
    s = filter_sentence.lower().strip()
    t = chunk_text.lower()
    ratio = fuzz.partial_ratio(s, t)
    keywords = [w for w in re.findall(r"[a-z0-9]+", s) if len(w) >= 3]
    hits = sum(1 for k in keywords if k in t)
    return int(ratio * 0.6 + hits * 10)


# -----------------------------
# LLM extraction
# -----------------------------

def build_system_prompt(allowed_categories: List[str]) -> str:
    return f"""You are a construction cost estimation extraction engine.
Extract system assemblies and line items from plain-text construction documents (notes/spec fragments).
Output must be STRICT JSON only, matching the provided schema.

Hard rules:
- Align each line item to MasterFormat/CSI as best as possible (Division + Section if inferable).
- Category MUST be one of: {allowed_categories}
- ONE LINE PER ITEM. Do not bundle multiple scopes into one item.
- Prefer system assemblies (e.g., Roof Assembly, Exterior Wall Assembly, HVAC System).
- Merge duplicates by providing stable merge_key fields (see schema).
- Include source.quote from the chunk to justify each item.
- If quantities are missing, use standard default dimensions and mark quantity.basis = "inferred_default_dimension".
- Do NOT output warnings. Do NOT output commentary. JSON only.
"""

def build_user_prompt(chunk: Chunk, region: str, measurement_standard: str, user_filter_sentence: Optional[str]) -> str:
    # Include the "when quantity is missing..." line explicitly as requested
    filter_clause = ""
    if user_filter_sentence and is_valid_filter(user_filter_sentence):
        filter_clause = f"""
Process ONLY scope relevant to this user filter (do not extract unrelated systems): {user_filter_sentence}
If nothing relevant exists, return empty assemblies list.
"""
    else:
        filter_clause = """
Process ALL systems found in this chunk.
"""

    return f"""
Region: {region}
Use measurement standard: {measurement_standard} (US => RSMeans; Caribbean/Commonwealth => NRM2/RICS).
When quantity is missing, refer to standard default dimensions for it.

{filter_clause}

Allowed categories: {ALLOWED_CATEGORIES}

Return JSON with this schema:
{{
  "document_meta": {{
    "project_name": "",
    "region": "{region}",
    "measurement_standard": "{measurement_standard}",
    "assumptions": []
  }},
  "assemblies": [
    {{
      "system_name": "",
      "system_type": "Architectural|Structural|MEP|Civil|Fire|Site|Other",
      "masterformat_division": "",
      "csi_section": null,
      "category": "",
      "items": [
        {{
          "div": "",
          "csi": null,
          "category": "",
          "item": "",
          "quantity": {{
            "value": null,
            "unit": null,
            "basis": "explicit|inferred_default_dimension|allowance|count_from_notes"
          }},
          "region_unit_standard": "{measurement_standard}",
          "confidence": 0.0,
          "source": {{
            "chunk_id": "{chunk.chunk_id}",
            "quote": "",
            "location_hint": null
          }},
          "merge_key": ""
        }}
      ],
      "assembly_merge_key": ""
    }}
  ]
}}

Chunk ID: {chunk.chunk_id}
Chunk text:
\"\"\"{chunk.text}\"\"\"
"""

def call_gpt_extract(client: OpenAI, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    out = resp.output_text.strip()
    print(f"GPT extract output: {out}")
    # Strict JSON parse
    return json.loads(out)


# -----------------------------
# Merging logic (assemblies + items)
# -----------------------------

def normalize_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def best_category(cat: str) -> str:
    if cat in ALLOWED_CATEGORIES:
        return cat
    return "Other"

def merge_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "document_meta": {
            "project_name": "",
            "region": "",
            "measurement_standard": "",
            "assumptions": []
        },
        "assemblies": []
    }

    assembly_map: Dict[str, Dict[str, Any]] = {}
    item_map: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (assembly_key, item_merge_key)

    for res in all_results:
        if not merged["document_meta"]["region"]:
            merged["document_meta"] = res.get("document_meta", merged["document_meta"])

        for a in res.get("assemblies", []):
            a_key = normalize_key(a.get("assembly_merge_key") or a.get("system_name") or "unknown_system")
            if a_key not in assembly_map:
                new_a = {
                    "system_name": a.get("system_name") or "Unknown System",
                    "system_type": a.get("system_type") or "Other",
                    "masterformat_division": a.get("masterformat_division") or "",
                    "csi_section": a.get("csi_section", None),
                    "category": best_category(a.get("category", "Other")),
                    "items": [],
                    "assembly_merge_key": a.get("assembly_merge_key") or a_key
                }
                assembly_map[a_key] = new_a

            for it in a.get("items", []):
                it["category"] = best_category(it.get("category", "Other"))
                it_merge = normalize_key(it.get("merge_key") or it.get("item") or "")
                if not it_merge:
                    continue
                k = (a_key, it_merge)

                if k not in item_map:
                    item_map[k] = it
                else:
                    # merge quantities if one is missing
                    existing = item_map[k]
                    q1 = existing.get("quantity", {}) or {}
                    q2 = it.get("quantity", {}) or {}
                    if (q1.get("value") is None) and (q2.get("value") is not None):
                        existing["quantity"] = q2
                    # keep higher confidence
                    if (it.get("confidence", 0) or 0) > (existing.get("confidence", 0) or 0):
                        existing["confidence"] = it.get("confidence", existing.get("confidence", 0))
                        existing["source"] = it.get("source", existing.get("source"))

    # rebuild assemblies with items
    for a_key, a in assembly_map.items():
        items = [v for (ak, _), v in item_map.items() if ak == a_key]
        # stable sort: div, csi, item
        items.sort(key=lambda x: (x.get("div") or "", x.get("csi") or "", normalize_key(x.get("item") or "")))
        a["items"] = items
        merged["assemblies"].append(a)

    # sort assemblies by name
    merged["assemblies"].sort(key=lambda a: normalize_key(a.get("system_name", "")))
    return merged


# -----------------------------
# Excel export (system name as row header)
# -----------------------------

def to_excel_rows(merged: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for a in merged.get("assemblies", []):
        system_name = a.get("system_name", "Unknown System")
        # system header row (system name in first column, rest blank)
        rows.append({
            "DIV": system_name,
            "CSI": "",
            "Category": "",
            "Item": "",
            "Quantity": "",
            "Unit": ""
        })

        for it in a.get("items", []):
            q = it.get("quantity", {}) or {}
            qv = q.get("value", None)
            qu = q.get("unit", None)
            rows.append({
                "DIV": it.get("div", ""),
                "CSI": it.get("csi", "") or "",
                "Category": it.get("category", ""),
                "Item": it.get("item", ""),
                "Quantity": "" if qv is None else qv,
                "Unit": "" if qu is None else qu
            })
    return pd.DataFrame(rows, columns=["DIV", "CSI", "Category", "Item", "Quantity", "Unit"])


# -----------------------------
# Main
# -----------------------------

def main(
    input_path: str = "input.txt",
    output_xlsx: str = "output.xlsx",
    region: str = "US",
    user_filter_sentence: Optional[str] = None
):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("[Step 1] Load input...")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    measurement_standard = measurement_standard_for_region(region)

    print("[Step 2] Chunk text...")
    chunks = make_chunks(text)
    print(f"  Total chunks: {len(chunks)}")
    with open("chunks.json", "w") as f:
        json.dump([{"chunk_id": ch.chunk_id, "start_char": ch.start_char, "end_char": ch.end_char, "text": ch.text} for ch in chunks], f, indent=2)

    valid_filter = is_valid_filter(user_filter_sentence or "")
    print("[Step 3] Filter validation...")
    print(f"  User filter provided: {bool(user_filter_sentence)}")
    print(f"  Filter is valid: {valid_filter}")
    if valid_filter:
        scored = []
        for ch in chunks:
            sc = filter_score(user_filter_sentence, ch.text)
            scored.append((sc, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [ch for sc, ch in scored if sc >= FILTER_SCORE_THRESHOLD]
        # if nothing meets threshold, process nothing (per your rule)
        chunks_to_process = selected
        print(f"  Chunks selected by filter: {len(chunks_to_process)} (threshold={FILTER_SCORE_THRESHOLD})")
    else:
        chunks_to_process = chunks
        print("  Processing all chunks (no valid filter).")

    sys_prompt = build_system_prompt(ALLOWED_CATEGORIES)

    print("[Step 4] Extract JSON per chunk...")
    per_chunk_results = []
    for i, ch in enumerate(chunks_to_process, 1):
        print(f"  - Processing {ch.chunk_id} ({i}/{len(chunks_to_process)}) tokens~{estimate_tokens(ch.text)}")
        user_prompt = build_user_prompt(ch, region, measurement_standard, user_filter_sentence)

        try:
            res = call_gpt_extract(client, sys_prompt, user_prompt)
            per_chunk_results.append(res)
            extracted_items = sum(len(a.get("items", [])) for a in res.get("assemblies", []))
            print(f"    Extracted assemblies: {len(res.get('assemblies', []))}, items: {extracted_items}")
        except Exception as e:
            print(f"    ERROR on {ch.chunk_id}: {e}")
            # continue without failing entire run
            continue

    with open("chunks_result.json", "w") as f:
        json.dump(per_chunk_results, f, indent=2)

    print("[Step 5] Merge similar system assemblies/items...")
    merged = merge_results(per_chunk_results)
    total_items = sum(len(a.get("items", [])) for a in merged.get("assemblies", []))
    print(f"  Merged assemblies: {len(merged.get('assemblies', []))}, merged items: {total_items}")

    print("[Step 6] Export to Excel (single sheet)...")
    df = to_excel_rows(merged)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Details")
    print(f"  Wrote: {output_xlsx}")
    print("[Done]")


if __name__ == "__main__":
    # Example usage:
    # main(input_path="input.txt", output_xlsx="output.xlsx", region="US", user_filter_sentence="Only roofing and roof insulation")
    main(
        input_path="normalize_text.txt",
        output_xlsx="output.xlsx",
        region=os.environ.get("PROJECT_REGION", "US"),
        user_filter_sentence=os.environ.get("USER_FILTER", "")
    )
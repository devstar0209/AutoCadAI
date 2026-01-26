import os
import re
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Optional token estimation
try:
    import tiktoken
except ImportError:
    tiktoken = None

from openai import OpenAI


# ----------------------------
# Configuration
# ----------------------------

DEFAULT_ALLOWED_CATEGORIES = [
    "Sitework",
    "Concrete",
    "Masonry",
    "Metals",
    "Wood/Plastics/Composites",
    "Thermal and Moisture Protection",
    "Openings",
    "Finishes",
    "Specialties",
    "Equipment",
    "Furnishings",
    "Special Construction",
    "Conveying Equipment",
    "Fire Suppression",
    "Plumbing",
    "HVAC",
    "Electrical",
    "Communications",
    "Electronic Safety and Security",
    "Integrated Automation",
    "General Requirements"
]

REGION_TO_STANDARD = {
    "US": "RSMeans_US",
    "USA": "RSMeans_US",
    "United States": "RSMeans_US",
    "Caribbean": "NRM2_RICS_Commonwealth",
    "Commonwealth": "NRM2_RICS_Commonwealth",
    "UK": "NRM2_RICS_Commonwealth",
    "UAE": "NRM2_RICS_Commonwealth",
    "Canada": "NRM2_RICS_Commonwealth",  # adjust if you use a different standard
}


# ----------------------------
# Utilities
# ----------------------------

def approx_token_count(text: str, model: str = "gpt-5.2") -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def slugify_system_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9|/ -]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def jaccard_sim(a: str, b: str) -> float:
    A = set(re.findall(r"[a-z0-9]+", a.lower()))
    B = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# ----------------------------
# Chunking
# ----------------------------

HEADING_PATTERNS = [
    r"^\s*GENERAL NOTES\s*$",
    r"^\s*SPECIAL NOTES\s*$",
    r"^\s*ABBREVIATIONS\s*$",
    r"^\s*LEGEND\s*$",
    r"^\s*FINISH(ES)? SCHEDULE\s*$",
    r"^\s*DOOR SCHEDULE\s*$",
    r"^\s*WINDOW SCHEDULE\s*$",
    r"^\s*SPECIFICATIONS\s*$",
    r"^\s*DIVISION\s+\d+.*$",
    r"^\s*Division\s+\d+.*$",
    r"^\s*SECTION\s+\d{2}\s?\d{2}\s?\d{2}.*$",
    r"^\s*\d{2}\s?\d{2}\s?\d{2}.*$"
]
HEADING_RE = re.compile("|".join(HEADING_PATTERNS), re.IGNORECASE | re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    headings: List[str]
    start_offset: int
    end_offset: int
    approx_tokens: int


def split_into_blocks(text: str) -> List[Tuple[str, int, int, List[str]]]:
    """
    Returns blocks as (block_text, start_offset, end_offset, headings_in_block)
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [(text, 0, len(text), [])]

    boundaries = [0] + [m.start() for m in matches] + [len(text)]
    blocks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        blk = text[start:end].strip()
        if not blk:
            continue
        headings = [m.group(0).strip() for m in matches if start <= m.start() < end]
        blocks.append((blk, start, end, headings))
    return blocks


def extract_global_notes(text: str, max_chars: int = 8000) -> str:
    # Pull GENERAL NOTES / SPECIAL NOTES sections if present
    notes = []
    for heading in ["GENERAL NOTES", "SPECIAL NOTES"]:
        m = re.search(rf"(?is)\b{heading}\b(.*?)(\n\s*\n|$)", text)
        if m:
            notes.append(f"{heading}:\n{m.group(1).strip()}")
    combined = "\n\n".join(notes).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[TRUNCATED]"
    return combined


def make_chunks(text: str,
                target_tokens: int = 10000,
                overlap_tokens: int = 1000) -> List[Chunk]:
    text = normalize_whitespace(text)
    blocks = split_into_blocks(text)

    chunks: List[Chunk] = []
    buffer = ""
    buf_start = None
    buf_headings = []
    buf_end = 0

    def flush_chunk(idx: int, carryover_text: str = ""):
        nonlocal buffer, buf_start, buf_headings, buf_end
        if not buffer.strip():
            return
        chunk_text = buffer.strip()
        cid = f"chunk_{idx:04d}"
        tok = approx_token_count(chunk_text)
        chunks.append(Chunk(
            chunk_id=cid,
            text=chunk_text,
            headings=list(dict.fromkeys(buf_headings)),
            start_offset=buf_start if buf_start is not None else 0,
            end_offset=buf_end,
            approx_tokens=tok
        ))
        buffer = carryover_text
        if buffer:
            # carryover start is approximate; keep previous end - len(carryover)
            buf_start = max(0, buf_end - len(carryover_text))
        else:
            buf_start = None
        buf_headings = []
        # buf_end stays as last end; will update on next append

    chunk_idx = 1
    for blk, start, end, headings in blocks:
        if buf_start is None:
            buf_start = start
        candidate = (buffer + "\n\n" + blk).strip() if buffer else blk
        cand_tokens = approx_token_count(candidate)

        if cand_tokens <= target_tokens:
            buffer = candidate
            buf_end = end
            buf_headings.extend(headings)
            continue

        # if buffer has content, flush with overlap
        if buffer:
            # create overlap from end of buffer
            overlap_text = buffer[-min(len(buffer), overlap_tokens * 4):]  # rough chars
            flush_chunk(chunk_idx, carryover_text=overlap_text)
            chunk_idx += 1

        # now add blk (might itself be huge)
        if approx_token_count(blk) <= target_tokens:
            buffer = blk
            buf_start = start
            buf_end = end
            buf_headings = headings[:]
        else:
            # hard split the block by paragraphs
            paras = re.split(r"\n\s*\n", blk)
            temp = ""
            temp_start = start
            temp_headings = headings[:]
            cur_offset = start
            for p in paras:
                p = p.strip()
                if not p:
                    continue
                cand = (temp + "\n\n" + p).strip() if temp else p
                if approx_token_count(cand) <= target_tokens:
                    temp = cand
                    cur_offset += len(p) + 2
                    continue
                # flush temp
                if temp:
                    cid = f"chunk_{chunk_idx:04d}"
                    chunks.append(Chunk(cid, temp, temp_headings, temp_start, min(end, cur_offset), approx_token_count(temp)))
                    chunk_idx += 1
                temp = p
                temp_start = cur_offset
                cur_offset += len(p) + 2
            if temp:
                cid = f"chunk_{chunk_idx:04d}"
                chunks.append(Chunk(cid, temp, temp_headings, temp_start, end, approx_token_count(temp)))
                chunk_idx += 1
            buffer = ""
            buf_start = None
            buf_headings = []
            buf_end = end

    flush_chunk(chunk_idx)
    return chunks


# ----------------------------
# Filter validation
# ----------------------------

def is_valid_system_filter(user_filter: Optional[str]) -> bool:
    """
    Simple deterministic validation: must be non-empty, not too long,
    and contain at least one alphabetic token.
    """
    if not user_filter:
        return False
    s = user_filter.strip()
    if len(s) < 3 or len(s) > 300:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    # reject obvious non-filter junk
    if re.fullmatch(r"(?i)(all|none|n/a|na)", s.strip()):
        return False
    return True


# ----------------------------
# LLM Calls
# ----------------------------

SYSTEM_PROMPT = """You are a construction cost estimation extraction engine.
Extract system-assembly line items from provided plain text notes/specs/schedules.
Hard requirements:
- Output STRICT JSON only (no markdown).
- One line per item (one JSON object per item in items array). Do NOT bundle multiple activities into one item.
- Align each item to MasterFormat/CSI (div and csi_code). If exact code not stated, infer best-fit.
- category MUST be one of allowed_categories provided by the user.
- Merge similar/duplicate assemblies by using a normalized system_name and stable system_key.
- Units must match the region standard (RSMeans_US or NRM2_RICS_Commonwealth).
- Include quantity; when quantity is missing, refer to standard default dimensions for it and state that in quantity_basis.
- Prefer measurable scope items (assemblies) over vague notes.
- Do not add warnings or commentary in output. JSON only.
"""

def build_user_prompt(chunk: Chunk,
                      global_notes: str,
                      allowed_categories: List[str],
                      region_standard: str,
                      user_filter: Optional[str],
                      filter_valid: bool) -> str:
    filter_instruction = ""
    if filter_valid:
        filter_instruction = f"""
User system filter (VALID): {user_filter}
Only extract items that match this filter (systems/assemblies relevant to it)."""
    else:
        filter_instruction = """
User system filter is INVALID or empty. Extract items for ALL systems."""

    return f"""
Region standard: {region_standard}
Allowed categories: {json.dumps(allowed_categories)}

Instruction: when quantity is missing, refer to standard default dimensions for it.

Chunk metadata:
- chunk_id: {chunk.chunk_id}
- headings_detected: {chunk.headings}

Global notes (may apply to all chunks):
{global_notes if global_notes else "[none]"}

{filter_instruction}

Now extract items from the following chunk text:
<<<CHUNK_TEXT_START
{chunk.text}
CHUNK_TEXT_END>>>

Return JSON with keys: chunk_id, region_standard, filtering, items, systems.
Ensure filtering.user_filter_valid and filtering.user_filter_applied are correct.
""".strip()


def call_gpt(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # You can tune these:
        temperature=0.1,
        max_output_tokens=8000,
    )
    text = resp.output_text.strip()
    return json.loads(text)


# ----------------------------
# Post-processing & merging
# ----------------------------

def normalize_item(item: Dict[str, Any], allowed_categories: List[str]) -> Dict[str, Any]:
    # enforce category allowed
    cat = item.get("category", "").strip()
    if cat not in allowed_categories:
        # fallback: map by division if possible
        cat = "General Requirements" if "01" in str(item.get("div", "")) else allowed_categories[0]
    item["category"] = cat

    # normalize fields
    for k in ["system_name", "system_key", "job_activity", "div", "csi_code", "unit", "quantity_basis", "source_ref"]:
        if k in item and isinstance(item[k], str):
            item[k] = item[k].strip()

    # ensure system_key exists
    if not item.get("system_key"):
        item["system_key"] = slugify_system_key(f"{item.get('category','')}|{item.get('system_name','')}")
    else:
        item["system_key"] = slugify_system_key(item["system_key"])

    # quantity numeric if possible
    q = item.get("quantity", "")
    if isinstance(q, str):
        qs = q.strip()
        try:
            item["quantity"] = float(qs)
        except Exception:
            # keep as-is
            item["quantity"] = qs

    # confidence
    try:
        item["confidence"] = float(item.get("confidence", 0.6))
    except Exception:
        item["confidence"] = 0.6

    return item


def merge_systems(items: List[Dict[str, Any]], sim_threshold: float = 0.82) -> List[Dict[str, Any]]:
    """
    Merge similar system assemblies by system_key similarity.
    We DO NOT merge line-items into one row; we only normalize system_name/system_key to avoid duplicate system labels.
    """
    canonical: List[Tuple[str, str]] = []  # (system_key, system_name)

    def find_canonical(key: str, name: str) -> Tuple[str, str]:
        best = None
        best_score = 0.0
        for ck, cn in canonical:
            score = max(jaccard_sim(key, ck), jaccard_sim(name, cn))
            if score > best_score:
                best_score = score
                best = (ck, cn)
        if best and best_score >= sim_threshold:
            return best
        canonical.append((key, name))
        return (key, name)

    for it in items:
        key = it.get("system_key", "")
        name = it.get("system_name", "")
        ck, cn = find_canonical(key, name)
        it["system_key"] = ck
        it["system_name"] = cn
    return items


# ----------------------------
# Excel export
# ----------------------------

def items_to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append({
            "System Name": it.get("system_name", ""),
            "DIV": it.get("div", ""),
            "CSI": it.get("csi_code", ""),
            "Category": it.get("category", ""),
            "Job activity": it.get("job_activity", ""),
            "Quantity": it.get("quantity", ""),
            "Unit": it.get("unit", ""),
            "Quantity basis": it.get("quantity_basis", ""),
            "Source": it.get("source_ref", "")
        })
    df = pd.DataFrame(rows)
    # Sort to group by system name (but still one sheet)
    df.sort_values(by=["System Name", "DIV", "CSI", "Category", "Job activity"], inplace=True, kind="stable")
    return df


# ----------------------------
# Main pipeline
# ----------------------------

def run_pipeline(
    input_txt_path: str,
    output_xlsx_path: str,
    region: str,
    allowed_categories: Optional[List[str]] = None,
    user_system_filter: Optional[str] = None,
    model: str = "gpt-5.2"
):
    allowed_categories = allowed_categories or DEFAULT_ALLOWED_CATEGORIES
    region_standard = REGION_TO_STANDARD.get(region, "RSMeans_US")

    with open(input_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    print("[Step 1] Normalize text...")
    raw_text = normalize_whitespace(raw_text)
    print(f"  chars={len(raw_text):,}, approx_tokens={approx_token_count(raw_text):,}")

    print("[Step 2] Extract global notes...")
    global_notes = extract_global_notes(raw_text)
    print(f"  global_notes_chars={len(global_notes):,}")

    print("[Step 3] Chunking...")
    chunks = make_chunks(raw_text, target_tokens=10000, overlap_tokens=1000)
    print(f"  chunks={len(chunks)}")
    print("  sample:", [(c.chunk_id, c.approx_tokens) for c in chunks[:3]])

    print("[Step 4] Validate user system filter...")
    filter_valid = is_valid_system_filter(user_system_filter)
    print(f"  filter_valid={filter_valid}, filter='{user_system_filter or ''}'")

    client = OpenAI()

    all_items: List[Dict[str, Any]] = []
    all_systems: Dict[str, set] = {}

    print("[Step 5] LLM extraction per chunk...")
    for i, chunk in enumerate(chunks, start=1):
        user_prompt = build_user_prompt(
            chunk=chunk,
            global_notes=global_notes,
            allowed_categories=allowed_categories,
            region_standard=region_standard,
            user_filter=user_system_filter,
            filter_valid=filter_valid
        )

        print(f"  - Processing {chunk.chunk_id} ({i}/{len(chunks)}), approx_tokens={chunk.approx_tokens}")
        try:
            data = call_gpt(client, model=model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        except Exception as e:
            print(f"    ERROR in {chunk.chunk_id}: {e}")
            continue

        items = data.get("items", []) or []
        systems = data.get("systems", []) or []

        print(f"    extracted_items={len(items)}, extracted_systems={len(systems)}")

        # Normalize items
        for it in items:
            it = normalize_item(it, allowed_categories)
            all_items.append(it)

        # Collect systems/aliases (debug use)
        for s in systems:
            sk = slugify_system_key(s.get("system_key", s.get("system_name", "")))
            all_systems.setdefault(sk, set()).update(s.get("aliases", []) if isinstance(s.get("aliases"), list) else [])

        # small pacing if needed
        time.sleep(0.2)

    print("[Step 6] Merge similar system assemblies (normalize system labels)...")
    before = len({it.get("system_key") for it in all_items})
    all_items = merge_systems(all_items, sim_threshold=0.82)
    after = len({it.get("system_key") for it in all_items})
    print(f"  unique_systems_before={before}, after={after}")
    print(f"  total_line_items={len(all_items)}")

    print("[Step 7] Export to Excel (single sheet)...")
    df = items_to_dataframe(all_items)
    with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Line Items", index=False)

    print(f"  wrote: {output_xlsx_path}")
    print("[Done]")


if __name__ == "__main__":
    # Edit these:
    INPUT_TXT = "normalize_text.txt"
    OUTPUT_XLSX = "extracted_line_items.xlsx"
    REGION = "US"  # e.g. "US" or "Caribbean"
    USER_SYSTEM_FILTER = None  # e.g. "HVAC ductwork and diffusers" or "" for invalid

    # If you have your own fixed allowed categories, put them here:
    ALLOWED_CATEGORIES = DEFAULT_ALLOWED_CATEGORIES

    run_pipeline(
        input_txt_path=INPUT_TXT,
        output_xlsx_path=OUTPUT_XLSX,
        region=REGION,
        allowed_categories=ALLOWED_CATEGORIES,
        user_system_filter=USER_SYSTEM_FILTER,
        model="gpt-5.2"
    )
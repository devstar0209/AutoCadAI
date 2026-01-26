import os
import re
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

from rapidfuzz import fuzz
import tiktoken

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Config
# ----------------------------
MODEL = "gpt-5.2"
TARGET_TOKENS = 1500
OVERLAP_LINES = 20

# If you have RSMeans locally, you can map lookup_query -> codes here.
RSMEANS_CODEBOOK = {
    # "tpo roof system fully adhered": {"assembly_code": "...", "line_item_code": "..."}
}

# ----------------------------
# Utilities
# ----------------------------
def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def normalize_text(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Preserve line structure; just normalize whitespace within lines
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in raw.split("\n")]
    # Remove excessive blank runs (keep at most 2)
    out = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out).strip() + "\n"

def add_line_numbers(text: str) -> str:
    lines = text.split("\n")
    numbered = []
    for i, ln in enumerate(lines, start=1):
        numbered.append(f"L{i:06d}: {ln}")
    return "\n".join(numbered)

def count_tokens(text: str, encoding_name="cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

# ----------------------------
# Chunking
# ----------------------------
HEADER_RE = re.compile(
    r"^\s*(L\d{6}:\s*)?(GENERAL NOTES|SPECIAL NOTES|SPECIFICATIONS|"
    r"DIVISION\s+\d+|SECTION\s+\d{2}\s*\d{2}\s*\d{2}|"
    r"[A-Z]-\d+|SHEET\s+\w+|ARCHITECTURAL|STRUCTURAL|MECHANICAL|ELECTRICAL|PLUMBING|CIVIL|FIRE)\b",
    re.IGNORECASE
)

BULLET_RE = re.compile(r"^\s*L\d{6}:\s*(?:\d+\.\s+|\([a-zA-Z0-9]+\)\s+|[-*]\s+)")
ALLCAPS_RE = re.compile(r"^\s*L\d{6}:\s*[A-Z0-9 ,/&()-]{8,}$")

@dataclass
class Chunk:
    chunk_id: str
    start_line: int
    end_line: int
    text: str

def split_into_coarse_sections(numbered_text: str) -> List[Tuple[int, int]]:
    lines = numbered_text.split("\n")
    header_lines = []
    for idx, ln in enumerate(lines, start=1):
        if HEADER_RE.search(ln):
            header_lines.append(idx)

    # Always include start
    if not header_lines or header_lines[0] != 1:
        header_lines = [1] + header_lines

    # Build ranges
    ranges = []
    for i in range(len(header_lines)):
        start = header_lines[i]
        end = (header_lines[i + 1] - 1) if i + 1 < len(header_lines) else len(lines)
        if end >= start:
            ranges.append((start, end))
    return ranges

def chunk_section(numbered_text: str, start_line: int, end_line: int) -> List[Chunk]:
    lines = numbered_text.split("\n")
    section_lines = lines[start_line - 1:end_line]

    # Build sub-block boundaries at bullets, all-caps headings, blank lines
    boundaries = [0]
    for i, ln in enumerate(section_lines):
        if i == 0:
            continue
        if BULLET_RE.match(ln) or ALLCAPS_RE.match(ln) or ln.strip() == "":
            # avoid too many splits on consecutive blanks
            boundaries.append(i)
    boundaries.append(len(section_lines))

    # Merge boundaries into token-sized chunks
    chunks: List[Chunk] = []
    cur_start = boundaries[0]
    cur_text_lines = []

    def flush(block_start_idx: int, block_end_idx: int):
        nonlocal chunks
        if block_end_idx <= block_start_idx:
            return
        block = section_lines[block_start_idx:block_end_idx]
        txt = "\n".join(block).strip()
        if not txt:
            return
        # Determine original line numbers from prefixed L000123:
        m1 = re.match(r"^\s*L(\d{6}):", block[0])
        m2 = re.match(r"^\s*L(\d{6}):", block[-1])
        s_ln = int(m1.group(1)) if m1 else start_line
        e_ln = int(m2.group(1)) if m2 else end_line
        chunks.append(Chunk(
            chunk_id=f"C{s_ln:06d}-{e_ln:06d}",
            start_line=s_ln,
            end_line=e_ln,
            text=txt
        ))

    # Accumulate blocks until hitting token target
    acc_start = 0
    acc_lines: List[str] = []
    acc_start_line = start_line

    for b in range(len(boundaries) - 1):
        b0, b1 = boundaries[b], boundaries[b + 1]
        block_lines = section_lines[b0:b1]
        candidate = "\n".join(acc_lines + block_lines)
        if count_tokens(candidate) > TARGET_TOKENS and acc_lines:
            # flush acc
            flush(acc_start, b0)
            # start new with overlap
            overlap_start = max(0, b0 - OVERLAP_LINES)
            acc_start = overlap_start
            acc_lines = section_lines[overlap_start:b1]
        else:
            if not acc_lines:
                acc_start = b0
            acc_lines.extend(block_lines)

    if acc_lines:
        flush(acc_start, len(section_lines))

    return chunks

def make_chunks(numbered_text: str) -> List[Chunk]:
    ranges = split_into_coarse_sections(numbered_text)
    all_chunks: List[Chunk] = []
    for (s, e) in ranges:
        all_chunks.extend(chunk_section(numbered_text, s, e))
    # De-dupe by chunk_id
    seen = set()
    deduped = []
    for ch in all_chunks:
        if ch.chunk_id not in seen:
            deduped.append(ch)
            seen.add(ch.chunk_id)
    return deduped

# ----------------------------
# GPT Extraction
# ----------------------------
SYSTEM_PROMPT = """You are a construction cost estimation extraction engine.
Extract system-level assemblies from text notes/specs and normalize them to CSI MasterFormat and RSMeans-style assemblies.

Rules:
- Output MUST be valid JSON only (no markdown).
- Prefer "system assemblies" (e.g., roof system, exterior wall system, slab-on-grade assembly, HVAC system) over individual materials.
- Capture constraints, inclusions/exclusions, and parameters (thickness, R-value, rating, finishes, etc).
- Use MasterFormat division/section where possible; include confidence 0-1.
- For RSMeans, provide a lookup_query and optionally codes if strongly supported. Include confidence 0-1.
- Quantities:
  - If explicit in text, set basis="EXPLICIT" and cite source_spans.
  - If quantity can be inferred from the text chunk alone, basis="INFERRED" with method.
  - If missing, set a placeholder quantity with basis="DEFAULT_REFERENCE" and explain the default_reference method.
  - Never fabricate project totals; if missing, prefer quantity=1 with unit appropriate for the assembly to carry unit cost.
- Always attach source_spans as line ranges (from the provided L000001 style prefixes).
- Identify conflicts (contradictory requirements) if present in this chunk.

JSON schema to follow:
{
  "chunk_id": string,
  "assemblies": [ ... ],
  "conflicts": [ ... ],
  "unclassified_items": [ ... ]
}
"""

def extract_json_from_chunk(client: OpenAI, chunk: Chunk) -> Dict[str, Any]:
    user_prompt = {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text
    }

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt)}
        ],
        # If your environment supports it, you can enforce JSON:
        # response_format={"type":"json_object"}
    )

    content = resp.output_text
    return json.loads(content)

# ----------------------------
# Merge / Reconcile
# ----------------------------
def normalize_key(a: Dict[str, Any]) -> str:
    mf = a.get("masterformat", {}) or {}
    div = (mf.get("division") or "").strip()
    sec = (mf.get("section") or "").strip()
    name = (a.get("system_name") or "").strip().lower()
    return f"{div}|{sec}|{name}"

def merge_assemblies(all_chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "assemblies": [],
        "conflicts": [],
        "unclassified_items": []
    }

    assemblies = []
    for r in all_chunk_results:
        assemblies.extend(r.get("assemblies", []))
        merged["conflicts"].extend(r.get("conflicts", []))
        merged["unclassified_items"].extend(r.get("unclassified_items", []))

    # De-dupe with fuzzy matching inside same MasterFormat section/division
    kept: List[Dict[str, Any]] = []
    for a in assemblies:
        a_key = normalize_key(a)
        best_idx = None
        best_score = 0
        for i, k in enumerate(kept):
            # compare only if same div/sec when available
            a_mf = a.get("masterformat", {}) or {}
            k_mf = k.get("masterformat", {}) or {}
            if (a_mf.get("division") and k_mf.get("division") and a_mf.get("division") != k_mf.get("division")):
                continue
            if (a_mf.get("section") and k_mf.get("section") and a_mf.get("section") != k_mf.get("section")):
                continue
            score = fuzz.token_set_ratio(a.get("system_name",""), k.get("system_name",""))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= 90:
            # merge fields (simple strategy: append lists, keep higher confidence)
            k = kept[best_idx]
            k.setdefault("source_spans", [])
            k["source_spans"].extend(a.get("source_spans", []))

            for list_field in ["constraints", "notes", "quantities", "allowances", "alternates"]:
                k.setdefault(list_field, [])
                k[list_field].extend(a.get(list_field, []))

            # choose best masterformat confidence
            if (a.get("masterformat", {}).get("confidence", 0) or 0) > (k.get("masterformat", {}).get("confidence", 0) or 0):
                k["masterformat"] = a.get("masterformat")

            # choose best rsmeans confidence
            if (a.get("rsmeans", {}).get("confidence", 0) or 0) > (k.get("rsmeans", {}).get("confidence", 0) or 0):
                k["rsmeans"] = a.get("rsmeans")

            # keep needs_human_review if either says so
            k.setdefault("quality", {})
            a_q = a.get("quality", {}) or {}
            k_q = k.get("quality", {}) or {}
            k_q["needs_human_review"] = bool(k_q.get("needs_human_review") or a_q.get("needs_human_review"))
            reasons = set((k_q.get("reasons") or []) + (a_q.get("reasons") or []))
            k_q["reasons"] = sorted(reasons)
            k["quality"] = k_q
        else:
            kept.append(a)

    # Assign assembly IDs
    for idx, a in enumerate(kept, start=1):
        a["assembly_id"] = a.get("assembly_id") or f"A-{idx:04d}"

        # Optional: apply RSMeans codebook mapping
        rs = a.get("rsmeans", {}) or {}
        q = (rs.get("lookup_query") or "").strip().lower()
        if q in RSMEANS_CODEBOOK:
            rs.update(RSMEANS_CODEBOOK[q])
            a["rsmeans"] = rs

    merged["assemblies"] = kept
    return merged

# ----------------------------
# Main
# ----------------------------
def run(input_path: str, output_path: str, project_id: str = "UNKNOWN") -> None:
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    norm = normalize_text(raw)
    numbered = add_line_numbers(norm)

    chunks = make_chunks(numbered)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    chunk_results = []
    for ch in chunks:
        res = extract_json_from_chunk(client, ch)
        chunk_results.append(res)

    merged = merge_assemblies(chunk_results)

    final = {
        "meta": {
            "project_id": project_id,
            "source": {
                "type": "text_dump",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "hash": file_hash(norm)
            },
            "model": MODEL,
            "units_default": {"length": "FT", "area": "SF", "volume": "CY", "count": "EA"}
        },
        "document_index": {
            "chunk_count": len(chunks),
            "chunk_ids": [c.chunk_id for c in chunks]
        },
        **merged,
        "assumptions_global": [
            "If quantities are missing, placeholders are provided with basis=DEFAULT_REFERENCE (typically quantity=1 in the appropriate unit) to support unit-costing; replace with takeoff quantities when available."
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to plain text dump file")
    ap.add_argument("--output", required=True, help="Path to output JSON")
    ap.add_argument("--project_id", default="UNKNOWN")
    args = ap.parse_args()

    run(args.input, args.output, args.project_id)
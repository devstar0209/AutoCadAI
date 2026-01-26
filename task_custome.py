import os
import re
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5.2"  # use the exact model name you have access to

MEP_FP_KEYWORDS = [
    # HVAC / controls
    "AHU", "RTU", "FCU", "VAV", "MAU", "DOAS", "ERV", "HRV", "EF", "SF",
    "DUCT", "DIFFUSER", "GRILLE", "REGISTER", "SMACNA", "TAB", "AIR BALANCE",
    "CHW", "HHW", "HW", "CW", "CONDENSATE", "REFRIG", "VRF", "MINI-SPLIT",
    "DDC", "BAS", "BMS", "CONTROLS", "SEQUENCE OF OPERATION",
    # Plumbing
    "DOMESTIC WATER", "SANITARY", "VENT", "STORM", "GAS PIPING", "BACKFLOW",
    "WATER HEATER", "PUMP", "PRV",
    # Fire protection
    "SPRINKLER", "STANDPIPE", "FDC", "FIRE PUMP", "NFPA", "RISER",
    # Electrical / low voltage / fire alarm
    "PANEL", "SWITCHGEAR", "TRANSFORMER", "CONDUIT", "FEEDER", "BRANCH CIRCUIT",
    "LIGHTING", "EMERGENCY", "GENERATOR", "ATS", "UPS",
    "FIRE ALARM", "FACP", "FA", "HORN", "STROBE", "SMOKE DETECTOR",
    "DATA", "COMM", "CCTV", "ACCESS CONTROL"
]

PAGE_BREAK_PATTERNS = [
    re.compile(r"^\s*(PAGE|SHEET)\s+[\w\.\-\/]+", re.IGNORECASE),
    re.compile(r"^\s*SECTION\s+\d{2}\s+\d{2}\s+\d{2}", re.IGNORECASE),
    re.compile(r"^\s*[A-Z]{1,3}\d{1,3}(\.\d+)?\s*$")  # standalone sheet id-like
]

HEADING_PATTERNS = [
    re.compile(r"^[A-Z0-9][A-Z0-9 \-\/]{6,}$"),  # ALL CAPS-ish
    re.compile(r".+:\s*$"),  # ends with colon
    re.compile(r"^\s*GENERAL NOTES\b", re.IGNORECASE),
    re.compile(r"^\s*SPECIAL NOTES\b", re.IGNORECASE),
    re.compile(r"^\s*MECHANICAL NOTES\b", re.IGNORECASE),
    re.compile(r"^\s*ELECTRICAL NOTES\b", re.IGNORECASE),
    re.compile(r"^\s*FIRE( PROTECTION)? NOTES\b", re.IGNORECASE),
]

BULLET_PATTERNS = [
    re.compile(r"^\s*[\-\u2022]\s+"),
    re.compile(r"^\s*\(?[0-9]+\)?[\.\)]\s+"),
    re.compile(r"^\s*\(?[A-Z]\)?[\.\)]\s+")
]

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_hint: Optional[str] = None
    sheet_hint: Optional[str] = None
    section_hint: Optional[str] = None
    tier: str = "B"  # A for MEP/FP-heavy
    char_start: int = 0
    char_end: int = 0

# ----------------------------
# Text preprocessing
# ----------------------------

def normalize_text(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Keep multiple newlines (useful for boundaries) but cap extremes:
    raw = re.sub(r"\n{4,}", "\n\n\n", raw)
    # Avoid destroying table spacing; do not collapse internal spaces aggressively.
    return raw

def is_page_break(line: str) -> bool:
    return any(p.search(line) for p in PAGE_BREAK_PATTERNS)

def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(p.search(s) for p in HEADING_PATTERNS)

def mep_fp_score(text: str) -> int:
    t = text.upper()
    score = 0
    for kw in MEP_FP_KEYWORDS:
        if kw in t:
            score += 1
    return score

def extract_hints(block_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    page = None
    sheet = None
    section = None

    m = re.search(r"\bPAGE\s+([A-Z0-9\.\-\/]+)\b", block_text, re.IGNORECASE)
    if m:
        page = f"PAGE {m.group(1)}"

    m = re.search(r"\bSHEET\s+([A-Z0-9\.\-\/]+)\b", block_text, re.IGNORECASE)
    if m:
        sheet = m.group(1)

    m = re.search(r"\bSECTION\s+(\d{2}\s+\d{2}\s+\d{2})\b", block_text, re.IGNORECASE)
    if m:
        section = m.group(1)

    # Also catch standalone sheet tokens like M1.1 / E2.0:
    m = re.search(r"\b([A-Z]{1,3}\d{1,3}(\.\d+)?)\b", block_text)
    if m and sheet is None:
        # only set if it looks like a drawing sheet reference
        sheet = m.group(1)

    return page, sheet, section

# ----------------------------
# Page -> block -> chunk
# ----------------------------

def split_into_pages(text: str) -> List[str]:
    lines = text.split("\n")
    pages = []
    cur = []
    for line in lines:
        if is_page_break(line) and cur:
            pages.append("\n".join(cur).strip("\n"))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        pages.append("\n".join(cur).strip("\n"))
    # If no meaningful page breaks detected, return single "page"
    if len(pages) == 1 and len(pages[0]) == len(text.strip("\n")):
        return [text]
    return pages

def split_page_into_blocks(page_text: str) -> List[str]:
    lines = page_text.split("\n")
    blocks = []
    cur = []
    for i, line in enumerate(lines):
        boundary = False
        if is_heading(line) and cur:
            boundary = True
        # Treat long blank gaps as boundaries
        if line.strip() == "" and cur and (i+1 < len(lines) and lines[i+1].strip() == ""):
            boundary = True

        if boundary:
            blocks.append("\n".join(cur).strip("\n"))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip("\n"))
    # Remove trivial blocks
    blocks = [b for b in blocks if len(b.strip()) > 0]
    return blocks

def pack_blocks_into_chunks(blocks: List[str],
                            max_chars: int,
                            overlap_chars: int,
                            tier: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf = ""
    buf_start = 0
    cursor = 0

    def flush(end_cursor: int):
        nonlocal buf, buf_start
        if buf.strip():
            cid = f"CH-{len(chunks):05d}"
            page_hint, sheet_hint, section_hint = extract_hints(buf)
            chunks.append(Chunk(
                chunk_id=cid,
                text=buf.strip(),
                page_hint=page_hint,
                sheet_hint=sheet_hint,
                section_hint=section_hint,
                tier=tier,
                char_start=buf_start,
                char_end=end_cursor
            ))
        buf = ""

    for b in blocks:
        b = b.strip("\n")
        if not b:
            continue

        # If a single block is huge, hard-split it with overlap
        if len(b) > max_chars:
            # flush existing buffer
            flush(cursor)
            # split this block
            start = 0
            while start < len(b):
                end = min(len(b), start + max_chars)
                piece = b[start:end]
                cid = f"CH-{len(chunks):05d}"
                page_hint, sheet_hint, section_hint = extract_hints(piece)
                chunks.append(Chunk(
                    chunk_id=cid,
                    text=piece.strip(),
                    page_hint=page_hint,
                    sheet_hint=sheet_hint,
                    section_hint=section_hint,
                    tier=tier,
                    char_start=cursor + start,
                    char_end=cursor + end
                ))
                if end >= len(b):
                    break
                start = max(0, end - overlap_chars)
            cursor += len(b) + 1
            buf_start = cursor
            continue

        # Try to append block to buffer
        candidate = (buf + "\n\n" + b) if buf else b
        if len(candidate) > max_chars and buf:
            flush(cursor)
            # start new buffer with overlap from previous chunk end
            # (character overlap within text is approximate; we also rely on GPT to handle continuity)
            buf = b
            buf_start = cursor
        else:
            buf = candidate

        cursor += len(b) + 2

    flush(cursor)
    return chunks

def chunk_text(text: str,
               max_chars_tier_a: int = 8000,
               max_chars_tier_b: int = 12000,
               overlap_chars: int = 800) -> List[Chunk]:
    pages = split_into_pages(text)
    all_chunks: List[Chunk] = []
    char_cursor = 0

    for p in pages:
        blocks = split_page_into_blocks(p)
        page_score = mep_fp_score(p)
        tier = "A" if page_score >= 4 else "B"
        max_chars = max_chars_tier_a if tier == "A" else max_chars_tier_b

        chunks = pack_blocks_into_chunks(
            blocks=blocks,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            tier=tier
        )

        # adjust char spans to global cursor
        for ch in chunks:
            ch.char_start += char_cursor
            ch.char_end += char_cursor
            all_chunks.append(ch)

        char_cursor += len(p) + 1

    # re-number chunk ids globally (after page loop)
    for i, ch in enumerate(all_chunks):
        ch.chunk_id = f"CH-{i:05d}"
    return all_chunks

# ----------------------------
# GPT extraction prompts
# ----------------------------

def build_system_prompt(user_custom_divisions: List[Dict[str, Any]]) -> str:
    return f"""You are an expert construction cost estimation extraction engine.
Extract MEP/FP scope first and with highest fidelity. Output MUST be valid JSON only.

Normalize findings into:
- systems (assemblies-first), components, and notes
- align CSI MasterFormat sections whenever possible
- include RSMeans hooks as code candidates (do not invent exact codes if not explicit)
- apply user custom divisions/tags when rules match; custom tags can override CSI grouping

Do NOT infer quantities unless explicitly stated. If not stated, use null and set basis="TBD".
Always include citations with chunk_id and an exact text_snippet.

User custom divisions:
{json.dumps(user_custom_divisions, indent=2)}
"""

def build_chunk_prompt(chunk: Chunk) -> str:
    return f"""Extract from this chunk.

Return JSON with keys:
- chunk_id
- page_hint, sheet_hint, section_hint
- mep_fp_priority: boolean
- global_notes: [] (only notes that apply broadly)
- systems: [] (system candidates, with assemblies/components if explicit)
- unclassified_findings: [] (anything relevant but not yet placeable)
- qa: open_questions/conflicts/assumptions (assumptions should be empty unless text explicitly states one)

Chunk metadata:
chunk_id={chunk.chunk_id}
tier={chunk.tier}
page_hint={chunk.page_hint}
sheet_hint={chunk.sheet_hint}
section_hint={chunk.section_hint}

CHUNK TEXT:
{chunk.text}
"""

# ----------------------------
# OpenAI call
# ----------------------------

def call_gpt_extract(client: OpenAI, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    # Uses Responses API style
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return json.loads(resp.output_text)

# ----------------------------
# Merge logic (simple, deterministic)
# ----------------------------

def merge_chunk_results(project_skeleton: Dict[str, Any], chunk_result: Dict[str, Any]) -> None:
    project_skeleton["global_notes"].extend(chunk_result.get("global_notes", []))
    project_skeleton["unclassified_findings"].extend(chunk_result.get("unclassified_findings", []))

    # Merge QA
    qa = chunk_result.get("qa", {})
    project_skeleton["qa"]["open_questions"].extend(qa.get("open_questions", []))
    project_skeleton["qa"]["assumptions"].extend(qa.get("assumptions", []))
    project_skeleton["qa"]["conflicts"].extend(qa.get("conflicts", []))

    # Merge systems (append; optional later dedupe pass)
    for sys in chunk_result.get("systems", []):
        project_skeleton["systems"].append(sys)

def build_project_skeleton(file_id: str, user_custom_divisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "project": {
            "name": "",
            "location": "",
            "source": {
                "file_id": file_id,
                "ingested_at": datetime.utcnow().isoformat() + "Z",
                "discipline_guess": [],
                "notes": ""
            }
        },
        "user_custom_divisions": user_custom_divisions,
        "extraction_meta": {
            "model": MODEL,
            "version": "1.0",
            "chunking": {}
        },
        "systems": [],
        "global_notes": [],
        "unclassified_findings": [],
        "qa": {
            "open_questions": [],
            "assumptions": [],
            "conflicts": []
        }
    }

# ----------------------------
# Optional: post-processing (system consolidation stub)
# ----------------------------

def consolidate_systems_naive(systems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Minimal consolidation: group by (system_name, system_type) when present.
    In production, do a second GPT pass to normalize/merge.
    """
    key_map = {}
    out = []
    for s in systems:
        name = (s.get("system_name") or "").strip().lower()
        stype = (s.get("system_type") or "").strip().lower()
        key = (name, stype)
        if name and key in key_map:
            idx = key_map[key]
            # shallow merges
            out[idx].setdefault("assemblies", []).extend(s.get("assemblies", []))
            out[idx].setdefault("components", []).extend(s.get("components", []))
            out[idx].setdefault("citations", []).extend(s.get("citations", []))
            out[idx].setdefault("masterformat", {}).setdefault("sections", [])
            out[idx]["masterformat"]["sections"] = sorted(set(
                out[idx]["masterformat"]["sections"] + s.get("masterformat", {}).get("sections", [])
            ))
        else:
            key_map[key] = len(out)
            out.append(s)
    return out

# ----------------------------
# Main
# ----------------------------

def main(
    input_txt_path: str,
    output_dir: str,
    user_custom_divisions_path: Optional[str] = None
):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    text = normalize_text(raw)
    file_id = os.path.basename(input_txt_path)

    user_custom_divisions = []
    if user_custom_divisions_path and os.path.exists(user_custom_divisions_path):
        with open(user_custom_divisions_path, "r", encoding="utf-8") as f:
            user_custom_divisions = json.load(f)

    chunks = chunk_text(text)
    print(f"Chunks created: {len(chunks)}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = build_system_prompt(user_custom_divisions)

    project = build_project_skeleton(file_id, user_custom_divisions)
    project["extraction_meta"]["chunking"] = {
        "strategy": "page->block->priority chunks",
        "max_chars_tier_a": 8000,
        "max_chars_tier_b": 12000,
        "overlap_chars": 800
    }

    jsonl_path = os.path.join(output_dir, "chunk_extractions.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as out_jsonl:
        for ch in chunks:
            prompt = build_chunk_prompt(ch)
            result = call_gpt_extract(client, system_prompt, prompt)

            # ensure chunk_id
            result["chunk_id"] = result.get("chunk_id") or ch.chunk_id
            out_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")

            merge_chunk_results(project, result)

    # naive consolidation (optional)
    project["systems"] = consolidate_systems_naive(project["systems"])

    out_path = os.path.join(output_dir, "project_extraction.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)

    print(f"Wrote:\n- {jsonl_path}\n- {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to continuous plain-text dump")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--user_divisions", required=False, help="Path to user custom divisions JSON")
    args = ap.parse_args()

    main(args.input, args.outdir, args.user_divisions)
import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import tiktoken
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5.2"  # change if your org uses a different name
TARGET_TOKENS = 1600
OVERLAP_TOKENS = 200
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.5

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Utilities: tokenization
# -----------------------------
def get_encoder(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("o200k_base")

ENC = get_encoder(MODEL)

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

# -----------------------------
# Step 1: detect blocks (notes/spec-like)
# -----------------------------
BLOCK_TITLE_RE = re.compile(
    r"^(?:PAGE\s+\d+|SHEET\s+[A-Z0-9\.\-]+|[A-Z][A-Z0-9\-\s\/]{3,80})$"
)

TYPE_HINTS = [
    ("general_notes", re.compile(r"\bGENERAL\s+NOTES\b", re.I)),
    ("special_notes", re.compile(r"\bSPECIAL\s+NOTES\b", re.I)),
    ("keynotes", re.compile(r"\bKEY\s*NOTES\b", re.I)),
    ("legend", re.compile(r"\bLEGEND\b", re.I)),
    ("schedule", re.compile(r"\bSCHEDULE\b", re.I)),
    ("spec_section", re.compile(r"^\s*(SECTION|DIVISION)\s+\d+", re.I)),
]

@dataclass
class Block:
    block_id: str
    title: str
    block_type: str
    start_char: int
    end_char: int
    page_hint: Optional[str]
    text: str

def guess_block_type(title: str, body: str) -> str:
    t = title or ""
    for typ, rx in TYPE_HINTS:
        if rx.search(t) or rx.search(body[:500]):
            return typ
    if "NOTE" in t.upper():
        return "special_notes"
    return "unknown"

def segment_into_blocks(full_text: str) -> List[Block]:
    lines = full_text.splitlines(keepends=True)
    blocks: List[Block] = []

    current_title = "DOCUMENT_START"
    current_start = 0
    current_buf = []
    cursor = 0
    page_hint = None

    def flush(end_cursor: int):
        nonlocal current_title, current_start, current_buf, page_hint
        if not current_buf:
            return
        body = "".join(current_buf).strip("\n")
        if not body.strip():
            current_buf = []
            current_start = end_cursor
            return
        btype = guess_block_type(current_title, body)
        block_id = f"blk_{len(blocks)+1:04d}"
        blocks.append(Block(
            block_id=block_id,
            title=current_title,
            block_type=btype,
            start_char=current_start,
            end_char=end_cursor,
            page_hint=page_hint,
            text=body
        ))
        current_buf = []
        current_start = end_cursor

    for line in lines:
        # page hint
        mpage = re.search(r"\bPAGE\s+(\d+)\b", line, re.I)
        if mpage:
            page_hint = mpage.group(1)

        stripped = line.strip()
        # treat strong headings as block boundaries
        if stripped and stripped == stripped.upper() and len(stripped) <= 80:
            # boundary if current_buf has content
            flush(cursor)
            current_title = stripped
        elif BLOCK_TITLE_RE.match(stripped) and len(stripped) <= 80 and stripped == stripped.upper():
            flush(cursor)
            current_title = stripped

        current_buf.append(line)
        cursor += len(line)

    flush(cursor)
    return blocks

# -----------------------------
# Step 2: build LLM chunks from blocks
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    block_ids: List[str]
    text: str
    start_char: int
    end_char: int

def chunk_blocks(blocks: List[Block],
                 target_tokens: int = TARGET_TOKENS,
                 overlap_tokens: int = OVERLAP_TOKENS) -> List[Chunk]:
    chunks: List[Chunk] = []
    cur_text_parts: List[str] = []
    cur_block_ids: List[str] = []
    cur_start = None
    cur_end = None

    def flush_chunk():
        nonlocal cur_text_parts, cur_block_ids, cur_start, cur_end
        if not cur_text_parts:
            return
        text = "\n\n".join(cur_text_parts).strip()
        if not text:
            return
        chunk_id = f"chk_{len(chunks)+1:04d}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            block_ids=list(cur_block_ids),
            text=text,
            start_char=cur_start if cur_start is not None else 0,
            end_char=cur_end if cur_end is not None else 0
        ))
        cur_text_parts = []
        cur_block_ids = []
        cur_start = None
        cur_end = None

    def add_overlap_from_previous(prev_text: str):
        # take last overlap_tokens of prev_text
        toks = ENC.encode(prev_text)
        if len(toks) <= overlap_tokens:
            return prev_text
        tail = toks[-overlap_tokens:]
        return ENC.decode(tail)

    for i, b in enumerate(blocks):
        b_text = f"[{b.block_id} | {b.block_type} | {b.title} | page={b.page_hint}]\n{b.text}".strip()
        if cur_start is None:
            cur_start = b.start_char
        cur_end = b.end_char

        tentative = ("\n\n".join(cur_text_parts + [b_text])).strip()
        if count_tokens(tentative) > target_tokens and cur_text_parts:
            # flush
            prev = "\n\n".join(cur_text_parts).strip()
            flush_chunk()

            # start new chunk with overlap from previous chunk + current block
            overlap_text = add_overlap_from_previous(prev)
            cur_text_parts = [f"[OVERLAP_FROM_PREVIOUS]\n{overlap_text}", b_text]
            cur_block_ids = ["OVERLAP"] + [b.block_id]
            cur_start = b.start_char
            cur_end = b.end_char
        else:
            cur_text_parts.append(b_text)
            cur_block_ids.append(b.block_id)

    flush_chunk()
    return chunks

# -----------------------------
# Step 3: LLM calls (multi-pass extraction)
# -----------------------------
SYSTEM_PROMPT = """You are an expert construction cost estimation assistant.
Extract system assemblies from plain-text drawings/spec notes with high traceability.
Do not invent details not present in the text. If missing, add to assumptions_needed or open_questions.
Always include evidence quotes with chunk_id and verbatim text snippets.
Return STRICT JSON only that matches the provided schema fragment.
"""

def call_llm_json(prompt: str) -> Dict[str, Any]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
            )
            text = resp.output_text
            return json.loads(text)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(SLEEP_BETWEEN * (attempt + 1))
    raise RuntimeError("LLM call failed unexpectedly.")

def prompt_pass1_requirements(chunk: Chunk) -> str:
    return f"""
You are given one chunk of project text.

TASK (PASS 1): Extract ONLY global requirements and constraints.
Focus on: codes/standards, coordination, tolerances, submittals, quality, testing, alternates/allowances, "by others".

Return JSON with keys:
{{
  "global_requirements": {{
    "codes_standards": [ ... ],
    "coordination_scope": [ ... ],
    "allowances_alternates": [ ... ]
  }},
  "open_questions": [ ... ]
}}

Chunk_id: {chunk.chunk_id}
TEXT:
{chunk.text}
""".strip()

def prompt_pass2_systems(chunk: Chunk) -> str:
    return f"""
TASK (PASS 2): Identify systems and assemblies mentioned or implied.
Output systems with assemblies stubs (no deep components yet unless explicitly stated).

Return JSON with keys:
{{
  "systems": [
    {{
      "system_id": "stable_id",
      "system_name": "...",
      "csi_division": "01|02|03|...|49|unknown",
      "scope_summary": "...",
      "assemblies": [
        {{
          "assembly_id": "stable_id",
          "assembly_name": "...",
          "classification": {{ "kind": "...", "subkind": null }},
          "typical_locations": ["..."],
          "performance": {{ "fire_rating_hr": null, "stc": null, "r_value": null, "acoustical_notes": null, "other": [] }},
          "components": [],
          "inclusions": [],
          "exclusions": [],
          "assumptions_needed": [],
          "confidence": 0.0,
          "evidence": [{{"chunk_id":"...", "quote":"...", "location_hint":null}}]
        }}
      ],
      "evidence": [{{"chunk_id":"...", "quote":"...", "location_hint":null}}]
    }}
  ],
  "open_questions": [ ... ]
}}

Rules:
- Use evidence quotes.
- If division not stated, infer cautiously or set "unknown".
- Create stable IDs: lowercase, underscores, include division when known.

Chunk_id: {chunk.chunk_id}
TEXT:
{chunk.text}
""".strip()

def prompt_pass3_assembly_details(chunk: Chunk, assembly_index: List[Dict[str, str]]) -> str:
    # assembly_index: list of {"assembly_id","assembly_name"} already found
    asm_list = "\n".join([f"- {a['assembly_id']}: {a['assembly_name']}" for a in assembly_index]) or "NONE"
    return f"""
TASK (PASS 3): Extract detailed components/requirements for assemblies ALREADY known.
Only fill details that appear in this chunk. Do not create new assemblies in this pass.

Known assemblies:
{asm_list}

Return JSON:
{{
  "assembly_updates": [
    {{
      "assembly_id": "...",
      "performance": {{ "fire_rating_hr": null|number, "stc": null|number, "r_value": null|number, "acoustical_notes": null|string, "other": ["..."] }},
      "components": [
        {{
          "component_id": "stable_id_or_null",
          "name": "...",
          "spec": {{ "size": null|string, "gauge": null|string, "spacing_in": null|number, "thickness": null|string }},
          "quantity_basis": {{ "uom": "SF|LF|EA|CY|TON|ALLOW", "basis_note": null|string }},
          "installation": {{ "key_requirements": ["..."], "by_others": false }},
          "evidence": [{{"chunk_id":"...", "quote":"...", "location_hint":null}}]
        }}
      ],
      "inclusions": ["..."],
      "exclusions": ["..."],
      "assumptions_needed": ["..."],
      "confidence_delta": -1.0|0.0|+1.0,
      "evidence": [{{"chunk_id":"...", "quote":"...", "location_hint":null}}]
    }}
  ],
  "open_questions": [ ... ]
}}

Chunk_id: {chunk.chunk_id}
TEXT:
{chunk.text}
""".strip()

# -----------------------------
# Step 4: merging logic
# -----------------------------
def merge_requirements(master: Dict[str, Any], part: Dict[str, Any]) -> None:
    mg = master.setdefault("global_requirements", {})
    pg = part.get("global_requirements", {})

    for k in ["codes_standards", "coordination_scope", "allowances_alternates"]:
        mg.setdefault(k, [])
        for item in pg.get(k, []) or []:
            mg[k].append(item)

def merge_open_questions(master: Dict[str, Any], part: Dict[str, Any]) -> None:
    master.setdefault("open_questions", [])
    for q in part.get("open_questions", []) or []:
        master["open_questions"].append(q)

def merge_systems(master: Dict[str, Any], part: Dict[str, Any]) -> None:
    master.setdefault("systems", [])
    sys_by_id = {s["system_id"]: s for s in master["systems"]}

    for s in part.get("systems", []) or []:
        sid = s["system_id"]
        if sid not in sys_by_id:
            master["systems"].append(s)
            sys_by_id[sid] = s
            continue

        ms = sys_by_id[sid]
        # merge shallow fields
        if not ms.get("scope_summary") and s.get("scope_summary"):
            ms["scope_summary"] = s["scope_summary"]
        ms.setdefault("evidence", [])
        ms["evidence"].extend(s.get("evidence", []) or [])

        # merge assemblies
        ms.setdefault("assemblies", [])
        asm_by_id = {a["assembly_id"]: a for a in ms["assemblies"]}
        for a in s.get("assemblies", []) or []:
            aid = a["assembly_id"]
            if aid not in asm_by_id:
                ms["assemblies"].append(a)
                asm_by_id[aid] = a
            else:
                ma = asm_by_id[aid]
                ma.setdefault("evidence", [])
                ma["evidence"].extend(a.get("evidence", []) or [])
                # do not overwrite detailed fields here; pass3 will update

def apply_assembly_updates(master: Dict[str, Any], updates: List[Dict[str, Any]]) -> None:
    # Build index
    asm_index: Dict[str, Dict[str, Any]] = {}
    for s in master.get("systems", []) or []:
        for a in s.get("assemblies", []) or []:
            asm_index[a["assembly_id"]] = a

    for up in updates or []:
        aid = up["assembly_id"]
        if aid not in asm_index:
            continue
        a = asm_index[aid]

        # performance merge
        a.setdefault("performance", {})
        perf = up.get("performance", {}) or {}
        for k, v in perf.items():
            if v is None:
                continue
            # keep first non-null or override? Here: override with new non-null
            a["performance"][k] = v

        # merge components append (dedupe by name+spec)
        a.setdefault("components", [])
        existing_keys = set()
        for c in a["components"]:
            existing_keys.add((c.get("name","").lower(), json.dumps(c.get("spec",{}), sort_keys=True)))

        for c in up.get("components", []) or []:
            key = (c.get("name","").lower(), json.dumps(c.get("spec",{}), sort_keys=True))
            if key not in existing_keys:
                a["components"].append(c)
                existing_keys.add(key)

        for field in ["inclusions", "exclusions", "assumptions_needed"]:
            a.setdefault(field, [])
            a[field].extend(up.get(field, []) or [])

        a.setdefault("evidence", [])
        a["evidence"].extend(up.get("evidence", []) or [])

        # confidence
        a["confidence"] = float(a.get("confidence", 0.0)) + float(up.get("confidence_delta", 0.0) or 0.0)
        a["confidence"] = max(0.0, min(1.0, a["confidence"]))  # clamp 0..1

# -----------------------------
# Main pipeline
# -----------------------------
def build_document_index(blocks: List[Block]) -> Dict[str, Any]:
    return {
        "pages_detected": sorted({b.page_hint for b in blocks if b.page_hint}),
        "blocks": [
            {
                "block_id": b.block_id,
                "block_type": b.block_type,
                "title": b.title,
                "page_hint": b.page_hint,
                "start_char": b.start_char,
                "end_char": b.end_char
            }
            for b in blocks
        ]
    }

def extract(full_text: str, source_id: str = "input_text") -> Dict[str, Any]:
    blocks = segment_into_blocks(full_text)
    chunks = chunk_blocks(blocks)

    master: Dict[str, Any] = {
        "meta": {
            "source_id": source_id,
            "extraction_model": MODEL,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunking": {
                "strategy": "page_section_semantic_token",
                "target_tokens": TARGET_TOKENS,
                "overlap_tokens": OVERLAP_TOKENS
            }
        },
        "document_index": build_document_index(blocks),
        "global_requirements": {
            "codes_standards": [],
            "coordination_scope": [],
            "allowances_alternates": []
        },
        "systems": [],
        "open_questions": []
    }

    # PASS 1 + PASS 2 per chunk
    for ch in chunks:
        p1 = call_llm_json(prompt_pass1_requirements(ch))
        merge_requirements(master, p1)
        merge_open_questions(master, p1)

        p2 = call_llm_json(prompt_pass2_systems(ch))
        merge_systems(master, p2)
        merge_open_questions(master, p2)

    # Build known assembly index for pass 3
    assembly_index = []
    for s in master.get("systems", []):
        for a in s.get("assemblies", []):
            assembly_index.append({"assembly_id": a["assembly_id"], "assembly_name": a.get("assembly_name", "")})

    # PASS 3 per chunk
    for ch in chunks:
        p3 = call_llm_json(prompt_pass3_assembly_details(ch, assembly_index))
        apply_assembly_updates(master, p3.get("assembly_updates", []))
        merge_open_questions(master, p3)

    return master

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Path to plain text file (all pages).")
    parser.add_argument("--outfile", required=True, help="Path to write extracted JSON.")
    parser.add_argument("--source-id", default="input_text")
    args = parser.parse_args()

    with open(args.infile, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    result = extract(full_text, source_id=args.source_id)

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
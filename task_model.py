import os
import re
import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

MODEL = "gpt-5.2"  # use your deployed GPT-5 model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PHASE = "CD"          # Concept | SD | DD | CD
MC_ITERATIONS = 10000

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    tier: str
    char_start: int
    char_end: int

# ============================================================
# TEXT NORMALIZATION & CHUNKING (SIMPLIFIED)
# ============================================================

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{4,}", "\n\n\n", text)

def chunk_text(text: str, max_chars=10000) -> List[Chunk]:
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(
            Chunk(
                chunk_id=f"CH-{i:05d}",
                text=text[start:end],
                tier="A",
                char_start=start,
                char_end=end
            )
        )
        start = end
        i += 1
    return chunks

# ============================================================
# GPT PROMPTS
# ============================================================

def build_extract_system_prompt():
    return """
You are a construction scope extraction engine.

Rules:
- Do NOT infer quantities
- Output valid JSON only
- Always include citations (chunk_id)
"""

def build_extract_user_prompt(chunk: Chunk):
    return f"""
Extract systems from this text.

chunk_id: {chunk.chunk_id}

TEXT:
{chunk.text}

Return JSON:
{{
  "systems": [],
  "notes": []
}}
"""

def build_cost_map_system_prompt():
    return """
You are a construction cost estimator specializing in RSMeans.

Rules:
- DO NOT invent RSMeans item numbers
- DO NOT invent prices
- Map systems to descriptive RSMeans-style assemblies
- Output JSON only
"""

def build_cost_map_user_prompt(system: Dict[str, Any]):
    return f"""
Map this system to RSMeans-style cost items:

{json.dumps(system, indent=2)}

Return JSON:
{{
  "assembly_cost_items": []
}}
"""

def build_uncertainty_prompt(item: Dict[str, Any]):
    return f"""
Classify uncertainty for this assembly.

Rules:
- Base ONLY on provided text
- If unclear, use LOW

Return JSON:
{{
  "quantity_certainty": "HIGH|MEDIUM|LOW",
  "spec_certainty": "HIGH|MEDIUM|LOW",
  "coordination_risk": true|false
}}

ASSEMBLY:
{json.dumps(item, indent=2)}
"""

# ============================================================
# OPENAI CALL
# ============================================================

def call_gpt(client, system_prompt, user_prompt):
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    text = resp.output_text
    return json.loads(text)

# ============================================================
# COST RANGE MODEL
# ============================================================

def compute_multipliers(phase, qty, spec, coord):
    base = {
        "Concept": (0.75, 1.0, 1.50),
        "SD": (0.85, 1.0, 1.35),
        "DD": (0.90, 1.0, 1.20),
        "CD": (0.95, 1.0, 1.10),
    }
    p10, p50, p90 = base.get(phase, base["SD"])

    if qty == "LOW":
        p10 -= 0.05
        p90 += 0.10
    if spec == "LOW":
        p90 += 0.10
    if coord:
        p90 += 0.10

    return {"p10": round(p10, 2), "p50": 1.0, "p90": round(p90, 2)}

def apply_range(base, m):
    return {
        "p10": round(base * m["p10"], 2),
        "p50": round(base * m["p50"], 2),
        "p90": round(base * m["p90"], 2),
    }

# ============================================================
# MONTE CARLO ENGINE
# ============================================================

def sample_triangular(p10, p50, p90):
    return random.triangular(p10, p90, p50)

def run_monte_carlo(items, iterations):
    totals = []
    for _ in range(iterations):
        total = 0
        for it in items:
            r = it["cost_range"]
            total += sample_triangular(r["p10"], r["p50"], r["p90"])
        totals.append(total)

    totals.sort()

    def pct(p):
        return round(totals[int(len(totals) * p)], 2)

    return {
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p80": pct(0.80),
        "p90": pct(0.90),
        "mean": round(statistics.mean(totals), 2),
        "std_dev": round(statistics.stdev(totals), 2)
    }

# ============================================================
# MAIN PIPELINE
# ============================================================

def main(input_txt, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_txt, "r", encoding="utf-8", errors="ignore") as f:
        raw = normalize_text(f.read())

    chunks = chunk_text(raw)
    client = OpenAI(api_key=OPENAI_API_KEY)

    project = {
        "systems": [],
        "cost_estimation": {
            "cost_ranges": {
                "items": []
            }
        }
    }

    # ---- EXTRACTION PASS ----
    for ch in chunks:
        res = call_gpt(
            client,
            build_extract_system_prompt(),
            build_extract_user_prompt(ch)
        )
        project["systems"].extend(res.get("systems", []))

    # ---- COST MAPPING + RANGE MODEL ----
    for sys in project["systems"]:
        mapped = call_gpt(
            client,
            build_cost_map_system_prompt(),
            build_cost_map_user_prompt(sys)
        )

        for asm in mapped.get("assembly_cost_items", []):
            # Placeholder base allowance (estimator editable)
            base_allowance = asm.get("base_allowance", 100000)

            uncertainty = call_gpt(
                client,
                "Classify uncertainty only. JSON only.",
                build_uncertainty_prompt(asm)
            )

            mult = compute_multipliers(
                PHASE,
                uncertainty["quantity_certainty"],
                uncertainty["spec_certainty"],
                uncertainty["coordination_risk"]
            )

            cost_range = apply_range(base_allowance, mult)

            project["cost_estimation"]["cost_ranges"]["items"].append({
                "system": sys.get("system_name"),
                "assembly": asm.get("assembly_name"),
                "base_allowance": base_allowance,
                "multipliers": mult,
                "cost_range": cost_range
            })

    # ---- MONTE CARLO ----
    mc = run_monte_carlo(
        project["cost_estimation"]["cost_ranges"]["items"],
        MC_ITERATIONS
    )

    project["cost_estimation"]["monte_carlo"] = {
        "iterations": MC_ITERATIONS,
        "distribution": "triangular",
        "results": mc
    }

    # ---- OUTPUT ----
    out_path = os.path.join(output_dir, "project_cost_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2)

    print(f"✔ Cost analysis written to {out_path}")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    main(args.input, args.outdir)

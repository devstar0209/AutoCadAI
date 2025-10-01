import os
import re
import json
import cv2
import pytesseract
from pytesseract import Output
import openai
import openpyxl
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# =================== CONFIG ===================
API_KEY = "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"
client = openai.OpenAI(api_key=API_KEY)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Linux
# pytesseract.pytesseract.tesseract_cmd = r"C:\Path\To\tesseract.exe"  # Windows
MAX_WORKERS = 4

# CSI division-based keywords
construction_keywords = {
    "03 - Concrete": ["concrete", "slab", "footing", "foundation", "column", "beam", "girder",
                      "pile", "pier", "rebar", "reinforcement", "formwork", "joint", "curing"],
    "04 - Masonry": ["masonry", "brick", "block", "cmu", "stone", "veneer", "grout", "mortar", "lintel"],
    "05 - Metals": ["steel", "weld", "bolt", "plate", "angle", "channel", "pipe", "tube", "joist", "deck"],
    "06 - Wood": ["wood", "lumber", "timber", "plywood", "osb", "truss", "joist", "stud", "sheathing"],
    "07 - Thermal & Moisture": ["roof", "roofing", "membrane", "insulation", "vapor barrier",
                                "sealant", "flashing", "shingle", "tile"],
    "08 - Openings": ["door", "window", "frame", "glazing", "curtain wall", "skylight"],
    "09 - Finishes": ["floor", "ceiling", "tile", "carpet", "paint", "coating", "plaster",
                      "gypsum", "drywall", "veneer", "paneling"],
    "21 - Fire Protection": ["sprinkler", "fire protection", "standpipe", "fire pump", "alarm"],
    "22 - Plumbing": ["plumbing", "pipe", "valve", "toilet", "sink", "water heater", "drainage", "fixture"],
    "23 - HVAC": ["hvac", "duct", "chiller", "boiler", "air handler", "diffuser", "damper", "ventilation"],
    "26 - Electrical": ["electrical", "conduit", "cable", "wire", "panel", "transformer", "lighting",
                        "outlet", "switch", "breaker", "generator", "feeder", "grounding", "data", "telecom"],
    "Measurement Units": ["dimension", "length", "width", "height", "depth", "elevation", "level", "slope",
                          "thickness", "diameter", "radius", "area", "volume", "square", "cubic", "linear",
                          "feet", "foot", "inch", "inches", "meter", "millimeter", "centimeter"]
}

# Regex pattern for conduit size extraction (matches e.g., 3/4”C, 1"C, 2”C, etc.)
CONDUIT_SIZE_REGEX = r'(\d+\/\d+|\d+(\.\d+)?)(”|")\s*C'

# =================== FRONTEND NOTIFY ===================
def notify_frontend(event_type, **kwargs):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "pdf_processing",
        {"type": event_type, **kwargs}
    )

# =================== OCR ===================
def extract_text_from_image(image_path: str) -> str:
    print(f"Entered OCR function: {image_path}")
    try:
        if not os.path.exists(image_path):
            return ""

        img = cv2.imread(image_path)
        if img is None:
            return ""
        print(f"Processing OCR function...: {image_path}")

        # Preprocess image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        denoised = cv2.medianBlur(enhanced, 3)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Get detailed OCR data with position information
        data = pytesseract.image_to_data(thresh, output_type=Output.DICT)

        # Group text by lines based on y-coordinate proximity
        items = []
        current_item = []
        last_y = None
        line_tolerance = 20  # pixels tolerance for same line

        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue

            y = data["top"][i]

            # If new text is close to previous (same line), group them
            if last_y is not None and abs(y - last_y) < line_tolerance:
                current_item.append(text)
            else:
                # Save current item if it exists
                if current_item:
                    items.append(" ".join(current_item))
                current_item = [text]

            last_y = y

        # Add the last item
        if current_item:
            items.append(" ".join(current_item))

        # Join all items with double newlines to separate different entries
        combined_text = "\n\n".join(items)

        # Clean up excessive whitespace
        combined_text = re.sub(r'\s+', ' ', combined_text)
        combined_text = re.sub(r'\n\s*\n', '\n\n', combined_text)

        return combined_text.strip()

    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return ""

# =================== AI COST ESTIMATION ===================

def validate_items_against_source_text(structured_data, source_text):
    """
    Universal validation to ensure structured items actually exist in the source text.
    Prevents AI hallucination across all construction categories.
    """
    if not structured_data or not source_text:
        return structured_data

    source_text_upper = source_text.upper()
    validated_data = {}

    for category, items in structured_data.items():
        validated_items = []

        for item in items:
            if isinstance(item, dict) and 'item' in item:
                item_name = item['item'].strip().upper()
                # Check for key words from the item name
                item_words = [word for word in item_name.split() if len(word) > 2]
                if item_words and any(word in source_text_upper for word in item_words):
                    validated_items.append(item)
                    print(f"✅ Validated item: {item['item']}")
                else:
                    print(f"⚠️ Filtered out non-existent item: {item['item']}")
                    continue

        if validated_items:
            validated_data[category] = validated_items

    print(f"📊 Universal validation complete: {len(validated_data)} categories validated")
    return validated_data

def should_use_nrm2(project_location=None, cad_text=""):
    """Determine if NRM2 standards should be used based on project characteristics"""

    # Caribbean countries that commonly use NRM2/RICS standards
    caribbean_countries = [
        "barbados", "trinidad", "tobago", "jamaica", "bahamas", "grenada",
        "st lucia", "dominica", "antigua", "barbuda", "st kitts", "nevis",
        "st vincent", "grenadines", "belize", "guyana", "suriname"
    ]

    # Commonwealth countries that may use NRM2
    commonwealth_indicators = [
        "commonwealth", "rics", "nrm2", "british", "uk standard",
        "metres", "cubic metres", "square metres"
    ]

    # Check project location
    if project_location:
        location_lower = project_location.lower()
        if any(country in location_lower for country in caribbean_countries):
            return True

    # Check CAD text for indicators
    # cad_lower = cad_text.lower()

    # # Look for metric units (strong indicator of NRM2)
    # metric_indicators = ["m³", "m²", "metres", "cubic metres", "square metres", "linear metres"]
    # if any(indicator in cad_lower for indicator in metric_indicators):
    #     return True

    # # Look for NRM2/RICS references
    # if any(indicator in cad_lower for indicator in commonwealth_indicators):
    #     return True

    # # Look for Caribbean location references in CAD text
    # if any(country in cad_lower for country in caribbean_countries):
    #     return True

    return False


def extract_project_location(cad_text):
    """Extract project location from CAD text to determine if NRM2 should be used"""
    if not cad_text:
        return None

    text_lower = cad_text.lower()

    # Caribbean countries and territories
    caribbean_locations = {
        "barbados": "Barbados",
        "trinidad": "Trinidad and Tobago",
        "tobago": "Trinidad and Tobago",
        "jamaica": "Jamaica",
        "bahamas": "Bahamas",
        "grenada": "Grenada",
        "st lucia": "Saint Lucia",
        "saint lucia": "Saint Lucia",
        "dominica": "Dominica",
        "antigua": "Antigua and Barbuda",
        "barbuda": "Antigua and Barbuda",
        "st kitts": "Saint Kitts and Nevis",
        "saint kitts": "Saint Kitts and Nevis",
        "nevis": "Saint Kitts and Nevis",
        "st vincent": "Saint Vincent and the Grenadines",
        "saint vincent": "Saint Vincent and the Grenadines",
        "grenadines": "Saint Vincent and the Grenadines",
        "belize": "Belize",
        "guyana": "Guyana",
        "suriname": "Suriname"
    }

    # Check for location indicators
    for location_key, location_name in caribbean_locations.items():
        if location_key in text_lower:
            return location_name

    # Check for other Commonwealth indicators
    commonwealth_indicators = ["commonwealth", "rics", "nrm2", "british standard"]
    for indicator in commonwealth_indicators:
        if indicator in text_lower:
            return "Commonwealth"

    return None


def get_construction_jobs(cad_text, project_location=None):
    print(f"Starting construction jobs analysis...")

    cad_text = preprocess_cad_text(cad_text)

    print(f"preprocess_cad_text: {cad_text}")

    # Determine if NRM2 standards should be used
    use_nrm2 = should_use_nrm2(project_location, cad_text)
    print(f"Using NRM2 standards: {use_nrm2}")

    # Build system prompt based on standards to use
    if use_nrm2:
        system_prompt = """You are a professional construction estimator with 20+ years of experience specializing in NRM2 (RICS) standards. Analyze CAD text and symbols to produce comprehensive, detailed cost estimates using NRM2 measurement principles.

1. NRM2 COMPLIANCE REQUIREMENTS:
   - Use NRM2 (New Rules of Measurement 2) - 2nd edition UK October 2021 - RICS standardized measurement rules
   - Apply net quantity measurement principles (measure to structural faces, exclude waste unless specified)
   - Follow RICS standard deduction rules for openings, overlaps, and double counting
   - Include 2024 MasterFormat (CSI) codes for international compatibility
   - Base costs on appropriate regional rates (Caribbean/Commonwealth markets)

2. DETAILED JOB ACTIVITY DESCRIPTIONS:
   Each Job Activity MUST include comprehensive specifications:
   - MATERIAL GRADE/TYPE: (e.g., "grade C25/30", "softwood stress graded C24", "PVC-U", "grade B500B")
   - DIMENSIONS/SIZES: (e.g., "300mm thick", "22mm diameter", "600x600mm", "2.5mm²")
        Apply default architectural and sitework/paving assumptions if dimensions are missing (see Tables below)
   - CONSTRUCTION METHOD: (e.g., "poured in-situ", "shop fabricated site erected", "bedded in cement mortar")
   - FINISH/QUALITY: (e.g., "smooth finish", "facework one side", "non-slip finish", "weather struck joint")
   - INCLUDED ITEMS: (e.g., "including vibrating and curing", "including ironmongery", "including terminations")

3. NRM2 WORK SECTIONS (MANDATORY):
   Use correct work section codes:
   - A10-A99: Preliminaries and General Conditions
   - D20-D99: Groundwork (excavation, filling, piling)
   - E10: In-situ concrete, E20: Formwork, E30: Reinforcement, E40: Precast concrete
   - F10: Brick/Block walling, F20: Natural stone, F30: Cast stone
   - G10: Structural steel, G20: Structural timber, G30: Metal decking
   - H10-H99: Cladding/Covering (curtain walling, sheeting, membranes)
   - H60: Roof coverings, H70: Rooflights
   - J10-J99: Waterproofing (liquid applied, sheet, cementitious)
   - K10-K99: Linings/Sheathing/Dry partitioning
   - L10: Windows/Rooflights, L20: Doors/Shutters, L30: Stairs/Walkways
   - M10-M99: Surface finishes (screeds, tiling, decorative papers, painting)
   - M20: Plastering/Rendering, M40: Stone/Concrete/Ceramic tiling, M50: Timber flooring
   - N10-N99: Furniture/Equipment
   - P10-P99: Building fabric sundries
   - Q10-Q99: Paving/Planting/Fencing/Site furniture
   - R10: Rainwater pipework/gutters, R11: Foul drainage above ground, R12: Drainage below ground
   - R13: Land drainage, R14: Sump and sewage pumping
   - S10-S99: Piped supply systems
   - S10: Cold water, S11: Hot water, S12: Central heating, S13: Compressed air, S14: Sanitary appliances/fittings
   - T10-T99: Mechanical heating/cooling/refrigeration systems
   - T30: Low temperature hot water heating, T31: Steam heating, T32: Warm air heating
   - T40: Cooling systems, T41: Chilled water, T42: Refrigeration
   - U10-U99: Ventilation/Air conditioning systems
   - U10: General ventilation, U11: Smoke extract/Smoke control, U12: Kitchen ventilation
   - U13: Car park ventilation, U14: Fume extract, U15: Air conditioning
   - V10-V99: Electrical supply/power/lighting systems
   - V10: Electrical generation plant, V11: HV supply/distribution, V12: LV supply/distribution
   - V13: Motors/Starters, V20: LV distribution, V21: General LV power, V22: General lighting
   - V23: Emergency lighting, V24: Specialist lighting, V25: Exit/Emergency signs
   - W10-W99: Communications/Security/Control systems
   - W10: Telecommunications, W20: Radio/TV/CCTV/Video, W21: Data transmission
   - W22: Audio systems, W23: Clocks, W30: Security systems, W40: Building management systems
   - X10-X99: Transport systems (lifts, escalators, moving walkways)
   - Y10-Y99: General fixings/supports/restraints
   - Z10-Z99: Simple building works incidental to landscape works

4. REQUIRED OUTPUT FORMAT:
   - EXACT JSON list only: [{"CSI code":"03 30 00","NRM2 Section":"E10","Category":"In-situ Concrete","Job Activity":"Reinforced concrete foundation, grade C25/30, poured in-situ, 300mm thick, including vibrating and curing","Quantity":25,"Unit":"m³","Rate":185.50,"Material Cost":2787.50,"Equipment Cost":464.00,"Labor Cost":1391.00,"Total Cost":4642.50}]
   - All fields mandatory; numeric values for costs/quantities/rate
   - "NRM2 Section" field with work section code (A10-Z99)
   - Units MUST be metric: m³, m², m, nr, kg, tonnes
   - Steel reinforcement Quantity and UNIT in Kilograms
   - Structural Steel to be Measured in "Tonnes"

5. DETAILED SPECIFICATION EXAMPLES:
   CONCRETE (E10): "Reinforced concrete foundation, grade C25/30, poured in-situ, 300mm thick, including vibrating and curing"
   MASONRY (F10): "Common brickwork, 215mm thick, in cement mortar 1:3, English bond, facework one side, pointed with weather struck joint"
   STEEL (G10): "Structural steel beams, grade S355, universal beam 356x171x67kg/m, shop fabricated, site erected with bolted connections"
   ELECTRICAL DISTRIBUTION (V20): "PVC insulated copper cable, 2.5mm², single core, drawn into 25mm PVC conduit, including terminations"
   ELECTRICAL POWER (V21): "13A socket outlets, twin switched with earth terminal, including back box and connections"
   ELECTRICAL LIGHTING (V22): "LED ceiling luminaires, 600x600mm, 36W, 4000K, including lamp, control gear and mounting accessories"
   PLUMBING (S10): "Copper pipes, 22mm diameter, table X, capillary fittings, including brackets and pipe insulation where required"
   HVAC (U15): "Air conditioning split unit, 5kW cooling capacity, wall mounted, including refrigerant pipework and controls"
   DRAINAGE (R12): "Vitrified clay drain pipes, 150mm diameter, flexible joints, laid in trenches including granular bedding"

6. COVERAGE AND NAME PRESERVATION (CRITICAL):
   - Use CAD_OCR_TEXT as ground truth. Do not omit major categories with strong signals.
   - PRESERVE EXACT ITEM NAMES from CAD_OCR_TEXT for named equipment/devices (NO sequential generation or pattern-based extrapolation).
   - CRITICAL: If CAD_OCR_TEXT contains "Panel EDL216-2", use ONLY "Panel EDL216-2" - do NOT create EDL216-3, EDL216-4, etc.
   - PROPAGATE ATTRIBUTES into the Job Activity text when present in CAD_OCR_TEXT:
       • rating (e.g., 125A, 30 kVA, 250 kW)
       • size (e.g., 22mm, 600x600mm, 2.5mm²)
       • length (from patterns like L=200, L=4", L=3.5m; treat as quantity + unit where possible)
   - PRESERVE MULTIPLICITY per distinct name/size/rating: separate rows and correct quantities for each distinct item.

   - For each drawing, you MUST extract and include ALL relevant construction elements if present, including but not limited to:
       • Foundations (strip, pad, raft, pile, etc.)
       • Walls (internal, external, retaining, block, brick, concrete, etc.)
       • Slabs (ground, upper, suspended, etc.)
       • Columns, Beams, Girders
       • Stairs, Walkstairs, Landings, Ramps
       • Roofs, Roof coverings, Rooflights
       • Doors, Windows, Openings
       • Finishes (floor, wall, ceiling, tiling, painting, etc.)
       • Sitework (paving, kerbs, landscaping, fencing, etc.)
       • Drainage, Plumbing, HVAC, Electrical, Fire Protection, Equipment
   - Do NOT skip or lose any items. If an item is present in the CAD_OCR_TEXT, it must be included in the output.
   - ANTI-HALLUCINATION: Use ONLY items that exist in CAD_OCR_TEXT - do not generate additional similar items.


6. COVERAGE AND PRESERVATION:
    - Use CAD_OCR_TEXT as ground truth
    - PRESERVE exact item names, ratings, sizes from CAD text (NO modifications or sequential generation)
    - Create separate line items for distinct specifications
    - Include ONLY items that actually exist in the CAD_OCR_TEXT
    - DO NOT generate similar items based on patterns (e.g., if EDL216-2 exists, do NOT create EDL216-3, EDL216-4)

7. DERIVATION RULES FOR MISSING ITEMS:
   - Branch cable (m) = (outlets + switches + fixtures) × 4m average
   - Feeder cable (m) = (panels + major equipment) × 60m average
   - Conduit (m) ≈ 9% of total cable length
   - Formwork (m²) = concrete contact area
   - Reinforcement = 80-120 kg per m³ of concrete
   - Apply Default Architectural and Sitework/Paving Assumptions only if CAD/OCR text is missing dimensions/specifications.

   **Default Architectural Assumptions Table**
   | Element | Typical / Default Size / Specification | Notes |
   |---------|----------------------------------------|-------|
   | Room Sizes | Bedroom (single): 9–12 m² | Floor area if missing |
   | | Bedroom (double): 12–18 m² | Floor area if missing |
   | | Living room: 12–30 m² | Small to large units |
   | | Dining room: 8–20 m² | Floor area if missing |
   | | Bathroom: 3–6 m² | Floor area if missing |
   | Ceiling Height | 2.4–2.7 m | Standard between finished floor and ceiling |
   | Wall Thickness | Internal walls: 100–150 mm | Brick/blockwork |
   | | External walls: 200–300 mm | Brick/blockwork or concrete |
   | Doors | Internal: 2.1 m × 0.8 m | Standard flush/panel |
   | | External: 2.1 m × 0.9 m | Includes frame |
   | Windows | Standard: 1.2 m × 1.0 m | Single/double glazing |
   | | Large / patio: 2.1 m × 1.8 m | Sliding/fixed |
   | Floor / Slab Thickness | Ground floor slab: 150–200 mm RC | Reinforced concrete incl. screed |
   | | Upper floor slab: 120–180 mm RC | Reinforced concrete incl. screed |
   | Stairs | Rise: 150–180 mm | Vertical height per step |
   | | Tread: 250–300 mm | Horizontal depth per step |
   | | Width: 900–1200 mm | Clear width of stair |
   | | Landings: 900–1200 mm | Minimum clear width/depth |
   | Doors & Openings Deduction | Deduct from wall/floor areas as per NRM2 | Only if specified in CAD/OCR |
   | Default Construction Method | Concrete: poured in-situ; masonry: bedded in cement mortar | Apply unless CAD text specifies otherwise |
   | Default Finish | Walls: smooth plaster/render; Floors: screed/tiled; Ceilings: painted | Apply unless CAD text specifies otherwise |

   **Default Sitework / Paving Assumptions Table**
   | Element | Typical / Default Size / Specification | Notes |
   |---------|----------------------------------------|-------|
   | Concrete Paving | 100–150 mm thick reinforced concrete slab | Standard pedestrian/light vehicle traffic |
   | Asphalt / Bituminous Pavement | 50–100 mm thick compacted asphalt | For driveways / parking areas |
   | Kerbs | 150 mm high × 200 mm wide | Standard concrete kerbs |
   | Footpaths / Walkways | 1.0–1.5 m width | Pedestrian access, concrete or pavers |
   | Driveways | 3–4 m width | Single lane, reinforced or compacted |
   | Site Furniture Spacing | Benches: 2–3 m apart; Trash bins: 15–20 m apart | Default if CAD does not specify |
   | Landscaping / Planting Beds | 1–3 m width | Shrubs or groundcover, default if CAD missing |
   | Fencing | 1.8–2 m high | Chain link, timber, or metal standard |
   | Paving Unit Sizes | Standard pavers: 200 × 100 × 60 mm | Modular concrete or clay pavers |
   | Expansion / Joint Spacing | Concrete paving: 4–6 m centers | Joints for shrinkage and thermal movement |

8. NRM2 COST STRUCTURE:
   - Total Cost = Material Cost + Labor Cost + Equipment Cost
   - Rate = Material Rate
   - If only the material rate is known, first calculate Total Cost and Material Cost as (Material Rate × Quantity), then derive Labor and Equipment costs using standard allocation percentages.
   - Material 60-70%, Labor 25-35%, Equipment 5-15%
   - Include regional adjustments for Caribbean markets

9. QUALITY VALIDATION:
   - Net quantities measured correctly
   - Appropriate work sections assigned
   - Detailed specifications included
   - Metric units consistently used
   - Cost relationships balanced
"""
    else:
        system_prompt = """You are a professional construction estimator with 20+ years of experience. Analyze CAD text and symbols to produce comprehensive cost estimates.

1. ACCURACY REQUIREMENTS:
   - Use 2024 MasterFormat (CSI) codes
   - Electrical labor: Use 2023–2024 NECA standards
   - Base costs on current US market rates (RSMeans-like, national averages)
   - Consider regional variation (use national average if unknown)
   - Quantities must be realistic for the described scope and scale

2. REQUIRED OUTPUT FORMAT:
   - EXACT JSON list only, no prose: [{"CSI code":"03 30 00","Category":"Concrete","Job Activity":"Cast-in-place Concrete Slab, 6-inch thick","Quantity":150,"Unit":"CY","Rate":125.5,"Material Cost":12000,"Equipment Cost":3000,"Labor Cost":4500,"Total Cost":19500}]
   - All fields are mandatory; values must be numbers for costs/quantities/rate
   
   - CSI Code MUST be organized in a progressive, six-digit sequence (with further specificity decimal extension, if required)
   - Job Activity MUST be specific and self-contained (type, size, capacity, method). Do NOT embed bracketed notes. Each activity is one line item

3. COVERAGE AND NAME PRESERVATION (CRITICAL):
    - Use CAD_OCR_TEXT as ground truth. Do not omit major categories with strong signals.
    - PRESERVE EXACT ITEM NAMES from CAD_OCR_TEXT for named equipment/devices (NO sequential generation or pattern-based extrapolation).
    - CRITICAL: If CAD_OCR_TEXT contains "Panel EDL216-2", use ONLY "Panel EDL216-2" - do NOT create EDL216-3, EDL216-4, etc.
    - PROPAGATE ATTRIBUTES into the Job Activity text when present in CAD_OCR_TEXT:
        • rating (e.g., 125A, 30 kVA, 250 kW)
        • size (e.g., 1”C, 1/2”C). Do NOT generate sequential or pattern-based conduit sizes (e.g., do not output 1”C, 1/2”C, 2”C, etc. unless each is explicitly present in the CAD_OCR_TEXT). Only use sizes that exist in the CAD_OCR_TEXT.
    - PRESERVE MULTIPLICITY per distinct name/size/rating: separate rows and correct quantities for each distinct item.  

5. QUANTITY/RATE/COST RULES (CRITICAL — ENFORCEMENT REQUIRED):
    - Default Cost Allocation (Required if breakdown unknown):
        • Material: 50–65%
        • Labor: 30–40%
        • Equipment: 5–15%
    - Total Cost Formula:
        • Material Cost = Rate * Quantity
        • Total Cost = Material Cost + Labor Cost + Equipment Cost
    
    - Market Source:
        • Use RSMeans 2024 national average installed costs as a reference baseline.
        • Electrical labor productivity and costs must follow NECA 2023–2024 standards.
    - OCR Rate Handling:
        • If a unit rate is found in OCR text but no labor/equipment split is provided, treat that number as a material cost only and calculate the missing labor and equipment portions according to typical US construction cost distribution.
        • If the OCR explicitly states that a rate is a total installed cost, split it according to the component bands above.
    - If only the material rate is known, first calculate Total Cost and Material Cost as (Material Rate × Quantity), then derive Labor and Equipment costs using standard allocation percentages.
6 SPECIFIC ATTENTION: Ensure these commonly missed items are captured if present:
   - CONCRETE / PAVING (slab-on-grade with 6\" default thickness if unspecified; CY or SF with conversion)
   - Transformers (note kVA ratings)
   - Any electrical panels with specific names
   - BRANCH CABLE: You MUST output a line item for branch cable if any RECEPTACLE, SWITCH, DIMMER, or LIGHT FIXTURE is present. Use 2#12 AWG + 1# 12 GRD. Calculate as: (number of RECEPTACLES + number of SWITCHES + number of DIMMERS + number of LIGHT FIXTURES) × 12 LF (use 25 LF if long runs are implied by the CAD text). Example: If there are 10 receptacles, 5 switches, and 8 light fixtures, Branch Cable = (10+5+8) × 12 = 276 LF.
   - FEEDER CABLE: You MUST output a line item for feeder cable if any PANEL, TRANSFORMER, or GENERATOR is present. Use 2#10 AWG + 1# 10 GRD. Calculate as: (number of PANELS + number of TRANSFORMERS + number of GENERATORS) × 200 LF. Example: If there are 2 panels and 1 transformer, Feeder Cable = (2+1) × 200 = 600 LF.
   - CONDUIT: Output only one line item for 3/4"C conduit, even if multiple branch or feeder cables are present. Use 3/4"C. Calculate as: 30% of the total cable length (branch cable length + feeder cable length). Example: If Branch Cable = 276 LF and Feeder Cable = 600 LF, total cable = 876 LF, so Conduit = 0.3 × 876 = 262.8 LF (round to nearest whole number). Do not repeat this item.
   
   - If the text includes any form of "paving" (e.g., "paving", "pavement", "PVG"),
        you must include it in the output. Do not omit paving items.
    - Classify paving as "Concrete" if it's concrete paving, otherwise "Sitework". always in SF, thickness stated or default 6". Include rebar 1.46 lbs to 2.81 lbs per SF seperately.
    - Do not skip or lose any items.
7. VALIDATION CHECKLIST:
    - Cover categories with strong signals (Concrete, Masonry, Metals, Finishes, Thermal/Moisture, HVAC, Plumbing, Electrical, Sitework) if present
    - Positive quantities/costs; math consistent; CSI format "XX XX XX"
    - PRESERVE exact item names, sizes, ratings, and multiplicities for ELECTRICAL equipment/devices/cables/conduit as implied by CAD_OCR_TEXT
    - ANTI-HALLUCINATION: Use ONLY items that exist in CAD_OCR_TEXT - do not generate additional similar items and sequantial items
"""

    user_prompt = f"""CAD_OCR_TEXT:
{cad_text}

REQUIRED ANALYSIS:
1) MANDATORY: Extract EVERY SINGLE construction item that appears in the CAD_OCR_TEXT. Do not skip or miss any items - be exhaustive.
2) CRITICAL: DO NOT generate similar or sequential names (e.g., if you see EDL216-2, do NOT create EDL216-3, EDL216-4, etc.)
3) DO NOT extrapolate or assume additional items based on patterns - ONLY extract what is explicitly mentioned.
4) Use EXACT item names as they appear in the drawings, but interpret symbols and abbreviations to meaningful construction terms.
5) Systematically scan for items in these categories (do not limit to major items - include all found):
   - CONCRETE: Foundations, slabs, footings, columns, beams, girders, piles, piers, rebar, reinforcement, formwork, joints, curing, CONCRETE PAVING, PAVING, sidewalks, driveways
   - MASONRY: Brick walls, CMU walls, block walls, stone, veneer, grout, mortar, lintels
   - METALS: Structural steel, beams, columns, stairs, handrails, joists, decking, welding, bolts, plates, angles, channels, pipes, tubes
   - WOOD: Lumber, timber, plywood, OSB, trusses, joists, studs, sheathing
   - THERMAL & MOISTURE: Roofing, roofing membrane, insulation, vapor barrier, sealant, flashing, shingles, tiles
   - OPENINGS: Doors, windows, frames, glazing, curtain walls, skylights
   - FINISHES: Flooring, ceilings, tile, carpet, paint, coating, plaster, gypsum, drywall, veneer, paneling
   - FIRE PROTECTION: Sprinklers, fire protection systems, standpipes, fire pumps, alarms
   - PLUMBING: Pipes, valves, toilets, sinks, water heaters, drainage, fixtures
   - HVAC: Ducts, chillers, boilers, air handlers, diffusers, dampers, ventilation, fans, AC units
   - ELECTRICAL: Conduit, cables, wires, panels, TRANSFORMERS (with kVA ratings), lighting, outlets, switches, breakers, GENERATOR, feeders, grounding, data, telecom, receptacles, DIMMER
   - SITEWORK: Excavation, grading, backfill, asphalt, sidewalks, curbs, landscaping
   - EQUIPMENT: Elevators, lifts
6) For each extracted item, create a detailed Job Activity line item with:
    - Exact item name (no generic renaming), includes rating/size when provided.
    - Preserves multiplicities per distinct name/size/rating (separate rows or quantities).
    - Assigns appropriate CSI code (or NRM2 Section if applicable), realistic quantity/unit, and computes Material/Labor/Equipment/Total costs with Rate.

OUTPUT: ONLY a valid JSON array as specified above, no text.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        response_text = response.choices[0].message.content
        print(f"Initial AI response received:: {response_text}")
        validated_response = validate_and_improve_response(response_text, cad_text)
        return validated_response
    except Exception as e:
        print(f"Error in get_construction_jobs: {e}")
        return get_construction_jobs_fallback(cad_text)


def validate_and_improve_response(response_text, cad_text):
    """Validate AI response and improve if necessary"""
    try:
        # Try to extract and parse JSON
        clean_json = extract_json_from_response(response_text)
        if not clean_json:
            print("No JSON found in response, attempting to extract...")
            # Try to find JSON without code blocks
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)

        if clean_json:
            parsed_data = json.loads(clean_json)

            # Validate the data structure and content
            validation_result = validate_construction_data(parsed_data)

            if validation_result["is_valid"]:
                print(f"Response validation passed: {validation_result['score']}/100")
                return parsed_data
            else:
                print(f"Response validation failed: {validation_result['errors']}")
                # Try to improve the response
                return improve_response_with_feedback(response_text, validation_result, cad_text)
        else:
            print("Could not extract valid JSON, using fallback...")
            return get_construction_jobs_fallback(cad_text)

    except Exception as e:
        print(f"Validation error: {e}")
        return get_construction_jobs_fallback(cad_text)

def validate_construction_data(data):
    """Validate construction data for accuracy and completeness"""
    errors = []
    score = 100

    if not isinstance(data, list):
        errors.append("Data is not a list")
        return {"is_valid": False, "errors": errors, "score": 0}

    if len(data) == 0:
        errors.append("No construction activities found")
        return {"is_valid": False, "errors": errors, "score": 0}

    required_fields = ["CSI code", "Category", "Job Activity", "Quantity", "Unit", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]
    # NRM2 Section is optional but validated if present
    optional_nrm2_fields = ["NRM2 Section"]

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not a dictionary")
            score -= 20
            continue

        # Check required fields
        for field in required_fields:
            if field not in item:
                errors.append(f"Item {i} missing field: {field}")
                score -= 5

        # Validate numeric fields
        numeric_fields = ["Quantity", "Rate", "Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]
        for field in numeric_fields:
            if field in item:
                try:
                    value = float(item[field])
                    if value <= 0:
                        errors.append(f"Item {i} {field} must be positive")
                        score -= 3
                except (ValueError, TypeError):
                    errors.append(f"Item {i} {field} is not a valid number")
                    score -= 5

        # Validate cost relationships
        if all(field in item for field in ["Material Cost", "Equipment Cost", "Labor Cost", "Total Cost"]):
            try:
                calculated_total = float(item["Material Cost"]) + float(item["Equipment Cost"]) + float(item["Labor Cost"])
                actual_total = float(item["Total Cost"])
                if abs(calculated_total - actual_total) > 0.01:  # Allow small rounding differences
                    errors.append(f"Item {i} total cost doesn't match sum of components")
                    score -= 10
            except (ValueError, TypeError):
                pass

        # Validate CSI code format (allow flexibility for NRM2 integration)
        if "CSI code" in item:
            csi_code = str(item["CSI code"])
            # Basic CSI format validation - allow some flexibility
            if not re.match(r'^\d{2}[\s\-]?\d{2}[\s\-]?\d{2}', csi_code):
                errors.append(f"Item {i} has invalid CSI code format: {csi_code}")
                score -= 3

        # Validate NRM2 section format if present
        if "NRM2 Section" in item:
            nrm2_section = str(item["NRM2 Section"])
            if not re.match(r'^[A-Z]\d{2}', nrm2_section):
                errors.append(f"Item {i} has invalid NRM2 section format: {nrm2_section}")
                score -= 3

        # Validate metric units for NRM2 compliance
        if "Unit" in item:
            unit = str(item["Unit"]).lower()
            metric_units = ["m³", "m²", "m", "nr", "kg", "tonnes", "m3", "m2"]
            imperial_units = ["cy", "sf", "lf", "ea", "tons", "cf"]

            # Check if using NRM2 (has NRM2 Section field or metric units)
            has_nrm2_section = "NRM2 Section" in item
            uses_metric = any(mu in unit for mu in metric_units)

            if has_nrm2_section and not uses_metric:
                errors.append(f"Item {i} has NRM2 section but non-metric unit: {unit}")
                score -= 2

    is_valid = score >= 70 and len(errors) <= 3
    return {"is_valid": is_valid, "errors": errors, "score": score}

def improve_response_with_feedback(original_response, validation_result, cad_text):
    """Improve AI response based on validation feedback"""
    feedback_prompt = f"""The previous response had these issues: {validation_result['errors']}

Please provide a corrected response that addresses these problems. Focus on:
1. Ensuring all required fields are present
2. Making sure all costs are positive numbers
3. Verifying that Total Cost = Material Cost + Equipment Cost + Labor Cost
4. Using proper CSI code format (XX XX XX)
5. Providing realistic quantities and costs

Original CAD text: {cad_text}

Return ONLY a valid JSON array with the exact structure specified in the system prompt."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional construction estimator. Fix the previous response based on the feedback provided."},
                {"role": "user", "content": feedback_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )

        improved_response = response.choices[0].message.content
        validated_response = validate_and_improve_response(improved_response, cad_text)

        print("Improved response generated based on feedback")
        return validated_response

    except Exception as e:
        print(f"Error improving response: {e}")
        return get_construction_jobs_fallback(cad_text)

def get_construction_jobs_fallback(cad_text):
    """Fallback method with simpler, more reliable approach"""
    print("Using fallback method for construction jobs...")

    fallback_prompt = f"""Based on this CAD drawing text, provide a simple list of common construction activities with basic cost estimates:

{cad_text}

Return a JSON array with this exact format:
[{{"CSI code": "03 30 00", "Category": "Concrete", "Job Activity": "Foundation", "Quantity": 100, "Unit": "CY", "Rate": 120, "Material Cost": 8000, "Equipment Cost": 2000, "Labor Cost": 2000, "Total Cost": 12000}}]

Include only the most obvious construction activities. Keep it simple and accurate."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a construction estimator. Provide simple, accurate cost estimates."},
                    {"role": "user", "content": fallback_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Fallback method also failed: {e}")
        # Return error message instead of default data
        return f'{{"error": "AI analysis failed: {str(e)}", "message": "Unable to process CAD drawing for cost estimation. Please check your OpenAI API key and try again."}}'

# =================== PDF PAGE PROCESSING ===================
def convert_pdf_page_to_image(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(pdf_path, dpi=200, first_page=page_number, last_page=page_number, poppler_path="/usr/bin")
    if not images: return ""
    directory = os.path.dirname(pdf_path)
    image_path = os.path.join(directory, f"page_{page_number}.png")
    images[0].save(image_path, "PNG")
    return image_path

# =================== OUTPUT GENERATION ===================
def generate_outputs(jobs_list: list, output_pdf: str, output_excel: str):
    print(f"Generating outputs...")
    headers = [
        "CSI code","Category","Job Activity","Quantity","Unit","Rate",
        "Material Cost","Equipment Cost","Labor Cost","Total Cost"
    ]
    table_data = [headers]

    material_sum = equipment_sum = labor_sum = total_sum = 0

    for item in jobs_list:
        # Safe parsing
        try:
            quantity = int(item.get("Quantity") or 0)
        except Exception:
            quantity = 0

        row = [
            str(item.get("CSI code","")),
            str(item.get("Category","")),
            str(item.get("Job Activity","")),
            f"{quantity:,}",
            str(item.get("Unit","")),
            f"${float(item.get('Rate') or 0):,.2f}",
            f"${float(item.get('Material Cost') or 0):,.2f}",
            f"${float(item.get('Equipment Cost') or 0):,.2f}",
            f"${float(item.get('Labor Cost') or 0):,.2f}",
            f"${float(item.get('Total Cost') or 0):,.2f}",
        ]
        table_data.append(row)

        material_sum += float(item.get('Material Cost') or 0)
        equipment_sum += float(item.get('Equipment Cost') or 0)
        labor_sum += float(item.get('Labor Cost') or 0)
        total_sum += float(item.get('Total Cost') or 0)

    # Safe summary row (auto matches header length)
    summary_row = [""] * (len(headers) - 5) + [
        "Total",
        f"${material_sum:,.2f}",
        f"${equipment_sum:,.2f}",
        f"${labor_sum:,.2f}",
        f"${total_sum:,.2f}"
    ]
    table_data.append(summary_row)

    # Validate row lengths & fix if needed
    expected_cols = len(headers)
    for i, row in enumerate(table_data):
        if len(row) < expected_cols:
            row.extend([""] * (expected_cols - len(row)))  # pad short rows
        elif len(row) > expected_cols:
            row = row[:expected_cols]  # truncate extras
            table_data[i] = row

    # Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=landscape(A3))
    table = Table(table_data, repeatRows=1)

    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,-1), (-1,-1), colors.lightgrey),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ])
    table.setStyle(style)

    doc.build([table])

    # Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Cost Estimation"
    for row in table_data:
        ws.append(row)
    wb.save(output_excel)

# =================== MAIN PDF PROCESSING WITH LIVE PROGRESS ===================
def get_page_count(pdf_file):
    reader = PdfReader(pdf_file)
    return len(reader.pages)
def start_pdf_processing(pdf_path: str, output_pdf: str, output_excel: str, location = None):
    total_pages = get_page_count(pdf_path)
    all_texts = [""] * total_pages

    def process_page(page_num):
        img_path = convert_pdf_page_to_image(pdf_path, page_num)
        if img_path:
            all_texts[page_num-1] = extract_text_from_image(img_path)
        progress = round((page_num / total_pages) * 100, 2)
        notify_frontend(
            "page_processed",
            page=page_num,
            total_pages=total_pages,
            progress=progress
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_page, range(1, total_pages + 1))

    combined_text = " ".join(all_texts)
    # Try to extract project location from PDF metadata or text
    # project_location = extract_project_location(combined_text)
    jobs_list = get_construction_jobs(combined_text, location)

    if jobs_list:
        generate_outputs(jobs_list, output_pdf, output_excel)
        notify_frontend(
            "pdf_processing_completed",
            pdf_path=output_pdf,
            excel_path=output_excel,
            progress=100
        )

def preprocess_cad_text(cad_text: str) -> str:
    """
    Preprocess CAD text to preserve multi-line item relationships while cleaning for AI processing.
    - Preserves line breaks for multi-line items
    - Groups related lines together
    - Cleans excessive whitespace while maintaining structure
    """

    # Split into lines and clean each line
    lines = cad_text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Remove excessive spaces but preserve single spaces
        line = re.sub(r'\s+', ' ', line.strip())
        # Skip empty lines
        if line:
            cleaned_lines.append(line.upper())

    # Join with newlines to preserve multi-line structure
    processed_text = '\n'.join(cleaned_lines)

    # Group related multi-line items more intelligently
    lines = processed_text.split('\n')
    grouped_items = []
    current_item_lines = []

    for i, line in enumerate(lines):
        # Check if this line looks like a continuation or a new item
        is_short_line = len(line.split()) <= 3  # Short lines might be continuations
        has_partial_word = not line.endswith(' ') and i < len(lines) - 1  # Line ends abruptly
        next_line_exists = i < len(lines) - 1

        # If current group exists and this line seems like a continuation
        if current_item_lines and (is_short_line or has_partial_word):
            current_item_lines.append(line)
        else:
            # Save current group if it exists
            if current_item_lines:
                grouped_items.append(' '.join(current_item_lines))
                current_item_lines = []

            # Start new group
            current_item_lines = [line]

    # Add the last group
    if current_item_lines:
        grouped_items.append(' '.join(current_item_lines))

    # Clean up the grouped items
    final_items = []
    for item in grouped_items:
        # Remove extra spaces and clean up
        item = re.sub(r'\s+', ' ', item.strip())
        if item:
            final_items.append(item)

    return '\n\n'.join(final_items)

def extract_json_from_response(response_text: str):
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match: return match.group(1)
    match2 = re.search(r'\[.*\]', response_text, re.DOTALL)
    return match2.group(0) if match2 else None

def validate_data(data: list) -> bool:
    return isinstance(data, list) and len(data) > 0

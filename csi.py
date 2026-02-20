import json

input_file = "resources_priced_jmd.json"
output_file = "resources_csi.json"

# Load original items
with open(input_file, "r", encoding="utf-8") as f:
    items = json.load(f)

filtered = []
remaining = []

for item in items:
    filtered.append({
        "item": item.get("item"),
        "CSI": item.get("CSI")
    })

# Write filtered (item + CSI only)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"Moved {len(filtered)} items to {output_file}")
import json

input_file = "resources_priced_jmd.json"
output_file = "resources_csi_1.json"

# Load original items
with open(input_file, "r", encoding="utf-8") as f:
    items = json.load(f)

filtered = []
remaining = []

for item in items:
    if item.get("unit", "").lower() == "hr":
        filtered.append({
            "item": item.get("item"),
            "CSI": item.get("CSI")
        })
    else:
        remaining.append(item)

# Write filtered (item + CSI only)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

# Overwrite input file with remaining items
with open(input_file, "w", encoding="utf-8") as f:
    json.dump(remaining, f, indent=2)

print(f"Moved {len(filtered)} M2 items to {output_file}")
print(f"Updated {input_file} with {len(remaining)} remaining items")
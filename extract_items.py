import json

INPUT_FILE = "resources_enriched.json"
OUTPUT_FILE = "resources_csi.json"

def extract_item_csi(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict-with-list cases
    if isinstance(data, dict):
        # try common container keys
        for key in ["items", "data", "records"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    result = []

    for obj in data:
        if not isinstance(obj, dict):
            continue

        result.append({
            "item": obj.get("item", ""),
            "CSI": obj.get("CSI", "")
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result)} records to {output_file}")


if __name__ == "__main__":
    extract_item_csi(INPUT_FILE, OUTPUT_FILE)

import json
import sys

def load_label_map(json_path):
    """
    Load label map from a JSON file for the YOLO model.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: List of labels.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Navigate to the labels list
        labels = data.get("mappings", {}).get("labels", [])
        
        if not labels:
            raise ValueError("No labels found in JSON file.")
        
        return labels

    except Exception as e:
        print(f"[ERROR] Failed to load label map: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_labels.py <path_to_json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    label_map = load_label_map(json_path)
    
    print("Extracted Labels:")
    for i, label in enumerate(label_map):
        print(f"{i}: {label}")

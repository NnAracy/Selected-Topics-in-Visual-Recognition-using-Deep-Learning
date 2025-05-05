import os
import json
from PIL import Image

data_root = "./hw3-data-release/"
test_dir = os.path.join(data_root, "test_release")
mapping_json = os.path.join(data_root, "test_image_name_to_ids.json")
output_json = os.path.join(data_root, "test.json")

with open(mapping_json, "r") as f:
    mapping = json.load(f)

images = []

for entry in mapping:
    file_name = entry["file_name"]
    image_id = entry["id"]
    img_path = os.path.join(test_dir, file_name)

    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping")
        continue

    try:
        with Image.open(img_path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

categories = [
    {"id": 1, "name": "class1"},
    {"id": 2, "name": "class2"},
    {"id": 3, "name": "class3"},
    {"id": 4, "name": "class4"}
]

test_json = {
    "images": images,
    "annotations": [],
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(test_json, f, indent=4)

print(f"{output_json} generated, total {len(images)} images")
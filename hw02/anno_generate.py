import os
import json
from PIL import Image

data_root = 'nycu-hw2-data/'
test_dir = os.path.join(data_root, 'test')

# Get list of image files sorted by filename number
image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
image_files.sort(key=lambda x: int(x.split('.')[0]))

# Generate image list
images = []
for idx, file_name in enumerate(image_files, start=1):
    img_path = os.path.join(test_dir, file_name)
    with Image.open(img_path) as img:
        width, height = img.size
    images.append({
        'id': idx,
        'file_name': file_name,
        'width': width,
        'height': height
    })

# Generate categories list
categories = [
    {'id': 1, 'name': '0'},
    {'id': 2, 'name': '1'},
    {'id': 3, 'name': '2'},
    {'id': 4, 'name': '3'},
    {'id': 5, 'name': '4'},
    {'id': 6, 'name': '5'},
    {'id': 7, 'name': '6'},
    {'id': 8, 'name': '7'},
    {'id': 9, 'name': '8'},
    {'id': 10, 'name': '9'}
]

# Construct JSON structure
test_json = {
    'images': images,
    'categories': categories,
    'annotations': []  # No annotations for test dataset
}

# Write JSON to file
with open(os.path.join(data_root, 'test.json'), 'w') as f:
    json.dump(test_json, f, indent=4)
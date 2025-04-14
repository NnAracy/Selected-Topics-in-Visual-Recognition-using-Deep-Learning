import json
import csv
import os
import argparse
from collections import defaultdict

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--method',
    type=str,
    choices=['wbf', 'nms', 'softnms'],
    default='wbf',
    help='Ensemble method'
)
args = parser.parse_args()

work_dir = f'./work_dirs/ensemble_{args.method}/'

# Load test set image IDs
test_json_path = './nycu-hw2-data/test.json'
with open(test_json_path, 'r') as f:
    test_data = json.load(f)
all_image_ids = {img['id'] for img in test_data['images']}

# Load prediction results
pred_json_path = os.path.join(work_dir, 'results.bbox.json')
with open(pred_json_path, 'r') as f:
    detections = json.load(f)

# Group detections by image_id
image_detections = defaultdict(list)
for det in detections:
    image_detections[det['image_id']].append(det)

results = []
for image_id in all_image_ids:
    dets = image_detections.get(image_id, [])

    if not dets:
        results.append((image_id, -1))
        continue

    dets.sort(key=lambda x: x['score'], reverse=True)

    for i in range(6):
        score_thr = 0.6 - i * 0.1
        valid_dets = [d for d in dets if d['score'] >= score_thr]
        if valid_dets:
            valid_dets.sort(key=lambda x: x['bbox'][0])
            final_digits = [str(d['category_id'] - 1) for d in valid_dets]
            break

    number = int(''.join(final_digits)) if valid_dets else -1
    results.append((image_id, number))

# Write to CSV
output_csv_path = os.path.join(work_dir, 'pred.csv')
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'pred_label'])
    for image_id, label in sorted(results, key=lambda x: x[0]):
        writer.writerow([image_id, label])

print(f"{len(results)} images solved, output to pred.csv")
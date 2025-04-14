import json
import os
import argparse
import numpy as np
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--method',
    type=str,
    choices=['wbf', 'nms', 'softnms'],
    default='wbf',
    help='Ensemble method'
)
parser.add_argument(
    '--iou-thr',
    type=float,
    default=0.55,
    help='IoU threshold for WBF/NMS'
)
parser.add_argument(
    '--skip-box-thr',
    type=float,
    default=0.001,
    help='Skip box threshold for WBF'
)
parser.add_argument(
    '--sigma',
    type=float,
    default=0.1,
    help='Sigma value for soft-NMS'
)
args = parser.parse_args()

# Paths
json_paths = [
    './work_dirs/carafe-faster-rcnn_r50_bl_bfp/results.bbox.json',
    './work_dirs/carafe-faster-rcnn_r101/results.bbox.json',
    './work_dirs/carafe-faster-rcnn_x101/results.bbox.json',
    './work_dirs/carafe-faster-rcnn_s50/results.bbox.json'
]
test_json_path = './nycu-hw2-data/test.json'
output_path = f'./work_dirs/ensemble_{args.method}/results.bbox.json'

# Load image info
with open(test_json_path, 'r') as f:
    test_data = json.load(f)
image_size_dict = {
    img['id']: (img['width'], img['height']) for img in test_data['images']
}

# Load all model detections
all_detections = []
for path in json_paths:
    with open(path, 'r') as f:
        all_detections.append(json.load(f))

image_ids = set()
for model_dets in all_detections:
    for det in model_dets:
        image_ids.add(det['image_id'])

# Fusion
fused_results = []

for image_id in sorted(image_ids):
    print(f"Processing image {image_id}")
    width, height = image_size_dict[image_id]

    boxes_list = []
    scores_list = []
    labels_list = []

    for model_dets in all_detections:
        boxes = []
        scores = []
        labels = []

        for det in model_dets:
            if det['image_id'] != image_id:
                continue
            x, y, w, h = det['bbox']
            x1 = max(0.0, min(1.0, x / width))
            y1 = max(0.0, min(1.0, y / height))
            x2 = max(0.0, min(1.0, (x + w) / width))
            y2 = max(0.0, min(1.0, (y + h) / height))
            boxes.append([x1, y1, x2, y2])
            scores.append(det['score'])
            labels.append(det['category_id'] - 1)

        boxes = np.array(boxes).reshape(-1, 4)
        scores = np.array(scores)
        labels = np.array(labels)

        if boxes.shape[0] == 0:
            continue

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if len(scores_list) == 0:
        continue

    # Perform Ensemble
    if args.method == 'wbf':
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
            conf_type='avg'
        )
    elif args.method == 'nms':
        boxes, scores, labels = nms(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=args.iou_thr
        )
    elif args.method == 'softnms':
        boxes, scores, labels = soft_nms(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=args.iou_thr,
            sigma=args.sigma,
            thresh=0.001
        )

    # Restore to original coordinates
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        fused_results.append({
            'image_id': image_id,
            'bbox': [
                round(x1 * width, 2),
                round(y1 * height, 2),
                round((x2 - x1) * width, 2),
                round((y2 - y1) * height, 2)
            ],
            'score': round(float(score), 6),
            'category_id': int(label) + 1
        })

# Output result
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(fused_results, f)

print(f"[✓] {args.method.upper()} ensemble result saved to {output_path}")
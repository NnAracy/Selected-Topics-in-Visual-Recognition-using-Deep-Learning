import json
import os
import cv2
import argparse
from collections import defaultdict

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image-id',
    type=int,
    default=None,
    help='image id to visualize (default: all test images)'
)
parser.add_argument(
    '--backbone',
    type=str,
    default='r50',
    choices=['r50', 'r101', 's50', 'x101'],
    help='backbone type'
)
parser.add_argument(
    '--score-thr',
    type=float,
    default=0.3,
    help='score threshold for bbox visualization'
)
args = parser.parse_args()

# Paths
data_root = './nycu-hw2-data'
test_dir = os.path.join(data_root, 'test')
test_json = os.path.join(data_root, 'test.json')

work_dir = f'./work_dirs/carafe-faster-rcnn_{args.backbone}'
pred_json = os.path.join(work_dir, 'results.bbox.json')
save_dir = os.path.join(work_dir, 'vis')
os.makedirs(save_dir, exist_ok=True)

# Load test image info
with open(test_json, 'r') as f:
    test_data = json.load(f)
img_id_to_filename = {
    img['id']: img['file_name'] for img in test_data['images']
}

# Load prediction results
with open(pred_json, 'r') as f:
    detections = json.load(f)

# Group by image ID
image_detections = defaultdict(list)
for det in detections:
    image_detections[det['image_id']].append(det)


# Color based on score range
def get_color_by_score(score):
    if score < 0.2:
        return (128, 128, 128)  # gray
    elif score < 0.4:
        return (255, 0, 0)      # blue
    elif score < 0.6:
        return (0, 255, 255)    # yellow
    elif score < 0.8:
        return (0, 165, 255)    # orange
    else:
        return (0, 0, 255)      # red


# Select image ids to visualize
if args.image_id is not None:
    image_ids = [args.image_id]
else:
    image_ids = list(image_detections.keys())

# Main loop
for image_id in image_ids:
    if image_id not in img_id_to_filename:
        print(f'[!] image_id {image_id} 不存在於 test.json，跳過')
        continue

    filename = img_id_to_filename[image_id]
    img_path = os.path.join(test_dir, filename)
    if not os.path.exists(img_path):
        print(f'[!] 圖片不存在: {img_path}')
        continue

    img = cv2.imread(img_path)
    dets = image_detections.get(image_id, [])

    for det in dets:
        score = det['score']
        if score < args.score_thr:
            continue

        x, y, w, h = det['bbox']
        label = det['category_id'] - 1
        color = get_color_by_score(score)

        # Draw bounding box
        cv2.rectangle(
            img,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            2
        )
        cv2.putText(
            img,
            f'{label}:{score:.2f}',
            (int(x), int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            color,
            1
        )

    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, img)
    print(f'[✓] image_id={image_id} visualization saved to {save_path}')
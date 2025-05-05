import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Instance segmentation prediction using MMDetection'
    )
    parser.add_argument('--config', required=True,
                        help='Path to the config file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to the checkpoint file')
    parser.add_argument('--output', default='test-results.json',
                        help='Path to save the output JSON file')
    parser.add_argument('--test-dir', default='hw3-data-release/test_release',
                        help='Directory of test images')
    parser.add_argument('--test-json', default='hw3-data-release/test.json',
                        help='Path to the test JSON file')
    parser.add_argument('--device', default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--score-thr', type=float, default=0.05,
                        help='Score threshold for filtering predictions')
    return parser.parse_args()


def main():
    args = parse_args()

    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device
    )

    with open(args.test_json, 'r') as f:
        test_info = json.load(f)

    final_results = []

    for img_info in tqdm(test_info['images'], desc='Running Inference'):
        img_path = os.path.join(args.test_dir, img_info['file_name'])
        image_id = img_info['id']

        result = inference_detector(model, img_path)
        instances = result.pred_instances.cpu().numpy()

        bboxes = instances.bboxes
        scores = instances.scores
        labels = instances.labels
        masks = instances.masks

        for i in range(len(bboxes)):
            score = float(scores[i])
            if score < args.score_thr:
                continue

            binary_mask = masks[i]
            rle = mask_utils.encode(
                np.asfortranarray(binary_mask.astype(np.uint8))
            )
            rle['counts'] = rle['counts'].decode('utf-8')

            final_results.append({
                'image_id': image_id,
                'category_id': int(labels[i]) + 1,
                'bbox': [float(x) for x in bboxes[i]],
                'score': score,
                'segmentation': rle
            })

    # === 輸出 JSON ===
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(final_results, f)

    print(f'Prediction result saved to: {args.output}')


if __name__ == '__main__':
    main()
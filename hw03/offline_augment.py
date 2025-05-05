import json
import uuid
import cv2
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.augmentations.geometric.transforms import ElasticTransform
from pycocotools import mask as mask_utils

ROOT = Path('hw3-data-release')
AUG_PER_IMG = 10
COPY_PROB = 1.0
MAX_TRY = 10
PATCH_ALPHA = 1
PATCH_SIGMA = 50


class SafeRandomCrop(A.DualTransform):
    def __init__(self, h, w, p=1.0):
        super().__init__(p)
        self.h, self.w = h, w

    def apply(self, img, top=0, left=0, **p):
        return img[top:top + self.h, left:left + self.w]

    def apply_to_mask(self, m, top=0, left=0, **p):
        return m[top:top + self.h, left:left + self.w]

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        h, w = img.shape[:2]
        pad_h, pad_w = max(self.h - h, 0), max(self.w - w, 0)

        if pad_h or pad_w:
            img = cv2.copyMakeBorder(
                img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
            params['image'] = img
            params['masks'] = [
                cv2.copyMakeBorder(
                    m, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
                )
                for m in params['masks']
            ]
            h, w = h + pad_h, w + pad_w

        return {
            'top': random.randint(0, h - self.h),
            'left': random.randint(0, w - self.w)
        }

    def get_transform_init_args_names(self):
        return ('h', 'w')


geom_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
    ElasticTransform(alpha=40, sigma=30, alpha_affine=0, p=0.5),
    SafeRandomCrop(512, 512, p=1.0)
])


def process_split(split):
    print(f'\n=== {split} (No-overlap) ===')
    src_dir = ROOT / split
    coco = json.load(open(ROOT / f'{split}.json'))

    # 建 patch 池 (class4)
    patches = []
    for a in coco['annotations']:
        if a['category_id'] != 4:
            continue
        info = next(i for i in coco['images'] if i['id'] == a['image_id'])
        img = cv2.imread(str(src_dir / info['file_name']))
        mask_full = mask_utils.decode(a['segmentation']).astype(np.uint8)
        ys, xs = np.where(mask_full)
        if xs.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        patches.append((
            img[y0:y1 + 1, x0:x1 + 1].copy(),
            mask_full[y0:y1 + 1, x0:x1 + 1]
        ))

    out_dir = ROOT / f'aug_{split}'
    out_dir.mkdir(exist_ok=True)
    images_out, ann_out = [], []
    img_id = 1

    def save(img, m_list, c_list):
        nonlocal img_id
        fname = f'{uuid.uuid4().hex}.jpg'
        cv2.imwrite(str(out_dir / fname), img)
        h, w = img.shape[:2]
        images_out.append({
            'id': img_id,
            'file_name': fname,
            'width': w,
            'height': h
        })
        for m, c in zip(m_list, c_list):
            if m.sum() == 0:
                continue
            rle = mask_utils.encode(np.asfortranarray(m))
            rle['counts'] = rle['counts'].decode()
            x, y, bw, bh = mask_utils.toBbox(rle)
            area = int(mask_utils.area(rle))
            ann_out.append({
                'id': len(ann_out) + 1,
                'image_id': img_id,
                'category_id': c,
                'segmentation': rle,
                'bbox': [float(x), float(y), float(bw), float(bh)],
                'area': area,
                'iscrowd': 0
            })
        img_id += 1

    by_img = {}
    for a in coco['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)

    for info in tqdm(coco['images']):
        base_img = cv2.imread(str(src_dir / info['file_name']))
        base_masks = [
            mask_utils.decode(a['segmentation']).astype(np.uint8)
            for a in by_img.get(info['id'], [])
        ]
        base_cats = [a['category_id'] for a in by_img.get(info['id'], [])]
        save(base_img, base_masks, base_cats)

        for _ in range(AUG_PER_IMG):
            ia = base_img.copy()
            ms = base_masks.copy()
            cs = base_cats.copy()

            if patches and random.random() < COPY_PROB:
                cell, pmask = random.choice(patches)
                ph, pw = pmask.shape
                ih, iw = ia.shape[:2]

                if ph < ih and pw < iw:
                    occupied = np.logical_or.reduce(ms) if ms else np.zeros(
                        (ih, iw), bool
                    )
                    for _ in range(MAX_TRY):
                        ty = random.randint(0, ih - ph)
                        tx = random.randint(0, iw - pw)
                        if np.any(occupied[ty:ty + ph, tx:tx + pw] & pmask):
                            continue
                        ia[ty:ty + ph, tx:tx + pw][pmask == 1] = cell[pmask == 1]
                        newm = np.zeros((ih, iw), np.uint8)
                        newm[ty:ty + ph, tx:tx + pw] = pmask
                        ms.append(newm)
                        cs.append(4)
                        break

            aug = geom_aug(image=ia, masks=ms)
            save(aug['image'], aug['masks'], cs)

    json.dump({
        'images': images_out,
        'annotations': ann_out,
        'categories': coco['categories']
    }, open(ROOT / f'aug_{split}.json', 'w'))

    print(f'✓ {split}: {len(images_out)} imgs')


if __name__ == '__main__':
    for sp in ['train', 'val']:
        process_split(sp)
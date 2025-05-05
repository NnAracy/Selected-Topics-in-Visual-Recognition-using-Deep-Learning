_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4),
        mask_head=dict(num_classes=4)
    )
)

dataset_type = 'CocoDataset'
classes = ('class1', 'class2', 'class3', 'class4')

data_root = './hw3-data-release/'

metainfo = {
    'classes': ('class1', 'class2', 'class3', 'class4'),
}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.05,
                p=0.4
            ),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=5,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
            dict(
                type='CLAHE',
                clip_limit=(1, 4),
                tile_grid_size=(8, 8),
                p=0.25
            ),
            dict(
                type='GaussNoise',
                var_limit=(10.0, 50.0),
                p=0.2
            ),
            dict(
                type='MotionBlur',
                blur_limit=3,
                p=0.15
            )
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels'],
            min_visibility=0.0,
            filter_lost_elements=True
        ),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True
    ),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='aug_train.json',
        data_prefix=dict(img='aug_train/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='aug_val.json',
        data_prefix=dict(img='aug_val/'),
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test_release/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'aug_val.json',
    metric=['bbox', 'segm'],
    format_only=False
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric=['bbox', 'segm'],
    format_only=True,
    outfile_prefix='./work_dirs/test'
)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

auto_scale_lr = dict(enable=True, base_batch_size=16)
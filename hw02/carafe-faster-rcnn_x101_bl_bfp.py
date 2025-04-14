_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
data_root = './nycu-hw2-data/'
test_dir = './work_dirs/carafe-faster-rcnn_x101/test'
dataset_type = 'CocoDataset'

model = dict(
    data_preprocessor=dict(pad_size_divisor=64),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnext101_64x4d'
        )
    ),
    neck=[
        dict(
            type='FPN_CARAFE',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            start_level=0,
            end_level=-1,
            norm_cfg=None,
            act_cfg=None,
            order=('conv', 'norm', 'act'),
            upsample_cfg=dict(  # implement carafe for upsampling
                type='carafe',
                up_kernel=5,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64
            )
        ),
        dict(
            type='BFP',  # use BFP to extract features
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local'
        )
    ],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.25, 0.5, 1.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            loss_bbox=dict(
                _delete_=True,
                type='BalancedL1Loss',  # use balanced L1 Loss in bbox head
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0
            )
        )
    )
)

metainfo = {
    'classes': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'palette': [
        (220, 20, 60), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]
}

backend_args = None
img_scale = (800, 400)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        size=img_scale,
        pad_val=dict(img=(114.0, 114.0, 114.0))
    ),
    dict(
        type='RandomAffine',
        max_rotate_degree=5.0,
        max_translate_ratio=0.05,
        scaling_ratio_range=(0.95, 1.05)
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='RandomErasing',
        n_patches=(1, 2),
        ratio=(0.05, 0.2)
    ),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline
    )
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 400), keep_ratio=True),
    dict(type='Pad', size=(800, 400), pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='PackDetInputs')
]

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid.json',
        data_prefix=dict(img='valid/'),
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(800, 400), keep_ratio=True),
            dict(type='Pad', size=(800, 400), pad_val=dict(img=114)),
            dict(type='PackDetInputs')
        ]
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric=['bbox'],
    format_only=True,
    outfile_prefix=test_dir
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid.json',
    metric=['bbox'],
    format_only=False
)

default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=3
    )
)

custom_hooks = [
    dict(
        type='SyncNormHook',
        priority=48
    ),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.001,
        update_buffers=True,
        priority=49
    )
]

auto_scale_lr = dict(enable=True, base_batch_size=16)
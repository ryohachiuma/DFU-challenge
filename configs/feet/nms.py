# dataset settings
_base_ = ['../faster_rcnn//faster_rcnn_x101_64x4d_fpn_1x_coco.py']
dataset_type = 'MyDataset'
data_root = './datasets/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
    
]

anno_file = 'groundtruth.csv'
#anno_file = 'part.cvv'

_dataset=dict(
        type='MyDataset',
        ann_file=[
                data_root + anno_file
        ],
        img_prefix=[data_root + 'images'],
        pipeline=train_pipeline)

test_dataset =dict(
        type='MyDataset',
        ann_file=[
                data_root + 'test.csv'
        ],
        img_prefix=[data_root + 'images'],
        pipeline=test_pipeline)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=_dataset,val=test_dataset)
evaluation = dict(interval=1, metric='mAP')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',init_kwargs=dict(project='mmdetection-tools', name='nms'))
    ])
train_cfg = dict(rcnn=dict(sampler=dict(type='OHEMSampler')))
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth'

total_epochs=50

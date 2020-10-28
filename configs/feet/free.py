# dataset settings
_base_ = ['../free_anchor//retinanet_free_anchor_x101_32x4d_fpn_1x_coco.py']
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
                data_root + 'groundtruth.csv'
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
        dict(type='WandbLoggerHook', init_kwargs=dict(project='mmdetection-tools', name='free'))
    ])

train_cfg = dict(rcnn=dict(sampler=dict(type='OHEMSampler')))
test_cfg = dict(
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_thr=0.5),
        max_per_img=100))
model = dict(bbox_head=dict(num_classes=1))
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth'
train_epochs=40



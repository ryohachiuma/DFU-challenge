# dataset settings
dataset_type = 'MyDataset'
data_root = '/home/ryo/ssd1/ryo/DFU/'
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MyDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'ground_truth.txt'
            ],
            img_prefix=[data_root + 'images'],
            pipeline=train_pipeline)),
    val=dict(
        type='MyDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'ground_truth.txt'
            ],
            img_prefix=[data_root + 'images'],
            pipeline=train_pipeline)),
    test=dict(
        type='MyDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'ground_truth.txt'
            ],
            img_prefix=[data_root + 'images'],
            pipeline=train_pipeline)))
evaluation = dict(interval=1, metric='mAP')

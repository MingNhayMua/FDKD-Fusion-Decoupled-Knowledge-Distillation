_base_ = [
    ''../_base_/datasets/tinyimagenet_bs64_224.py', '../_base_/default_runtime.py', '../_base_/schedules/imagenet_bs256.py''
]

teacher_ckpt = 'checkpoints/swinb_fully/best.pth'

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='mmpretrain.ImgDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    architecture=dict(
        type='mmpretrain.ImageClassifier',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
        ),
        data_preprocessor=dict(
            type='mmpretrain.ClsDataPreprocessor',
            num_classes=200,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        backbone=dict(
            type='mmpretrain.ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='mmpretrain.GlobalAveragePooling'),
        head=dict(
            type='mmpretrain.LinearClsHead',
            num_classes=200,
            in_channels=512,
            loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))),
    teacher=dict(
        type='mmpretrain.ImageClassifier',
        data_preprocessor=dict(
            type='mmpretrain.ClsDataPreprocessor',
            num_classes=200,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        backbone=dict(
            type='mmpretrain.SwinTransformer',
            arch='base',
            drop_path_rate=0.1,
            img_size=224),
        neck=dict(type='mmpretrain.GlobalAveragePooling'),
        head=dict(
            type='mmpretrain.LinearClsHead',
            num_classes=200,
            in_channels=1024,
            loss=dict(type='mmpretrain.LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
            topk=(1, 5)),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0),
        ]),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            feat=dict(type='ModuleInputs', source='head.fc'),
            data_samples=dict(type='ModuleInputs', source='')),
        teacher_recorders=dict(
            feat=dict(type='ModuleInputs', source='head.fc')),
        distill_losses=dict(
            loss_crd=dict(type='CRDLoss', loss_weight=0.8)),
        connectors=dict(
            # Student: 512-d → 128-d embedding
            loss_crd_stu=dict(type='CRDConnector', dim_in=512, dim_out=128),
            # Teacher: 1024-d → 128-d embedding
            loss_crd_tea=dict(type='CRDConnector', dim_in=1024, dim_out=128)),
        loss_forward_mappings=dict(
            loss_crd=dict(
                s_feats=dict(
                    from_student=True,
                    recorder='feat',
                    data_idx=0,
                    connector='loss_crd_stu'),
                t_feats=dict(
                    from_student=False,
                    recorder='feat',
                    data_idx=0,
                    connector='loss_crd_tea'),
                data_samples=dict(
                    from_student=True, recorder='data_samples', data_idx=1)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
work_dir = 'work_dirs/$(basename)'

custom_imports = dict(imports=['mmpretrain', 'mmrazor'], allow_failed_imports=False)

# ---- CRD requires PackCRDClsInputs and CRDDataset wrapper ----
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(_scope_='mmrazor', type='PackCRDClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(_scope_='mmrazor', type='PackCRDClsInputs'),
]

# Original Tiny-ImageNet datasets
ori_train_dataset = dict(
    _scope_='mmpretrain',
    type='CustomDataset',
    data_root='data/tiny-imagenet-200',
    data_prefix='train',
    pipeline=train_pipeline)

crd_train_dataset = dict(
    _scope_='mmrazor',
    type='CRDDataset',
    dataset=ori_train_dataset,
    neg_num=16384,
    sample_mode='exact',
    percent=1.0)

ori_test_dataset = dict(
    _scope_='mmpretrain',
    type='CustomDataset',
    data_root='data/tiny-imagenet-200',
    data_prefix='val',
    pipeline=test_pipeline)

crd_test_dataset = dict(
    _scope_='mmrazor',
    type='CRDDataset',
    dataset=ori_test_dataset,
    neg_num=16384,
    sample_mode='exact',
    percent=1.0)

train_dataloader = dict(
    _delete_=True,
    batch_size=64,
    num_workers=8,
    dataset=crd_train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=8,
    dataset=crd_test_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

test_dataloader = val_dataloader

# Auto scale learning rate based on actual batch size
auto_scale_lr = dict(base_batch_size=128)

# Add gradient clipping to prevent NaN loss
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5.0))

# Add warmup scheduler to stabilize early training
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[30, 60, 90],
        gamma=0.1)
]

# Override base config's load_from (doesn't work with distillation wrapper)
load_from = None

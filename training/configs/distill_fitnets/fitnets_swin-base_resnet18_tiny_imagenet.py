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
        # Record pre-logit features (input to head.fc) and logits (output of head.fc)
        student_recorders=dict(
            feat=dict(type='ModuleInputs', source='head.fc'),
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            feat=dict(type='ModuleInputs', source='head.fc'),
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            # FitNets hint loss: L2 on pre-logit features (student 512-d → teacher 1024-d)
            loss_feat=dict(type='L2Loss', loss_weight=10),
            # Logit-level KL divergence
            loss_kl=dict(
                type='KLDivergence', tau=6, loss_weight=10, reduction='mean')),
        connectors=dict(
            # Project student features (512) to teacher dimension (1024)
            loss_feat_sfeat=dict(
                type='TorchNNConnector',
                module_name='Linear',
                module_args=dict(in_features=512, out_features=1024))),
        loss_forward_mappings=dict(
            loss_feat=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='feat',
                    data_idx=0,
                    connector='loss_feat_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='feat',
                    data_idx=0)),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
work_dir = 'work_dirs/$(basename)'

custom_imports = dict(imports=['mmpretrain', 'mmrazor'], allow_failed_imports=False)

# ---- Augmentation (same as DKD Swin-Base → ResNet18 config) ----
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(batch_size=64, num_workers=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=32, num_workers=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(batch_size=32, num_workers=8, dataset=dict(pipeline=test_pipeline))

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

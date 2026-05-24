_base_ = [
    ''../_base_/datasets/tinyimagenet_bs64_224.py', '../_base_/default_runtime.py', '../_base_/schedules/imagenet_bs256.py''
]

teacher_ckpt = 'work_dirs/$(basename)'

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
            type='mmpretrain.ResNet',
            depth=101,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='mmpretrain.GlobalAveragePooling'),
        head=dict(
            type='mmpretrain.LinearClsHead',
            num_classes=200,
            in_channels=2048,
            loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc'),
            gt_labels=dict(type='ModuleInputs', source='head.loss_module')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_dkd=dict(
                type='DKDLoss',
                tau=1,
                alpha=1.0,
                beta=0.5,
                loss_weight=1,
                reduction='mean')),
        loss_forward_mappings=dict(
            loss_dkd=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc'),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
work_dir = 'work_dirs/$(basename)'

custom_imports = dict(imports=['mmpretrain', 'mmrazor'], allow_failed_imports=False)

train_dataloader = dict(batch_size=128, num_workers=8)
val_dataloader = dict(batch_size=64, num_workers=8)
test_dataloader = dict(batch_size=64, num_workers=8)


_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'
# =====================================================================
# Jet Detection YOLOv8-S Config (F16, F18, F22, F35)
# v2 — Optimized: Daha yüksek çözünürlük, MixUp, güçlü augmentation,
#       daha sık validation, AMP, batch shapes enabled
# =====================================================================

class_name = ('F16', 'F18', 'F22', 'F35')
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# Dataset yolları (train script tarafından override edilebilir)
data_root = './'
train_ann_file = 'coco_annotations/instances_train.json'
val_ann_file = 'coco_annotations/instances_validation.json'
train_data_prefix = 'archive/dataset/'
val_data_prefix = 'archive/dataset/'

# =====================================================================
# EĞİTİM PARAMETRELERİ
# =====================================================================
max_epochs = 150          # 120→150: daha uzun eğitim, CosineAnnealing ile tamamlanır
close_mosaic_epochs = 20  # Son 20 epoch Mosaic kapalı (fine-tune)
train_batch_size_per_gpu = 4
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2
base_lr = 0.0008          # 0.001→0.0008: 4-class küçük dataset için optimize

# Pretrained COCO ağırlıkları
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'

# =====================================================================
# GÖRÜNTÜ ÇÖZÜNÜRLÜĞÜ — 640→800 (küçük nesne tespiti için kritik)
# =====================================================================
img_scale = (800, 800)  # 640→800: jet uçakları için daha iyi detay

# =====================================================================
# MODEL
# =====================================================================
model = dict(
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        assigner=dict(
            num_classes=num_classes,
            topk=13,      # 10→13: daha fazla pos sample, küçük dataset'te etkili
        )),
    # NMS iyileştirmesi
    test_cfg=dict(
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),  # 0.7→0.65: daha agresif NMS
        max_per_img=300))

# =====================================================================
# DATA — Augmentation Pipeline (Güçlendirilmiş)
# =====================================================================

# Albumentations — daha agresif augmentation
albu_train_transforms = [
    dict(type='Blur', p=0.03),
    dict(type='MedianBlur', p=0.03),
    dict(type='ToGray', p=0.02),
    dict(type='CLAHE', p=0.03),
    # Yeni: Jet uçakları çeşitli hava koşullarında görünür
    dict(type='RandomBrightnessContrast', brightness_limit=0.2,
         contrast_limit=0.2, p=0.15),
    dict(type='HueSaturationValue', hue_shift_limit=10,
         sat_shift_limit=30, val_shift_limit=20, p=0.1),
    dict(type='ImageCompression', quality_lower=75, quality_upper=100, p=0.05),
]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

# Stage 1: Mosaic + MixUp (güçlü augmentation)
train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    # MixUp eklendi — 2 farklı Mosaic'li görüntüyü birleştirir
    dict(
        type='YOLOv5MixUp',
        prob=0.15,
        pre_transform=[
            *pre_transform,
            dict(
                type='Mosaic',
                img_scale=img_scale,
                pad_val=114.0,
                pre_transform=pre_transform),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(0.5, 1.5),
                max_aspect_ratio=100,
                border=(-img_scale[0] // 2, -img_scale[1] // 2),
                border_val=(114, 114, 114)),
        ]),
    *last_transform
]

# Stage 2: Son epoch'larda Mosaic/MixUp kapalı
train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    *last_transform
]

# =====================================================================
# DATALOADERS
# =====================================================================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        pipeline=train_pipeline))

# Batch shapes: val'de +0.02 mAP kazandırır
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='coco_annotations/instances_test.json',
        data_prefix=dict(img=val_data_prefix),
        batch_shapes_cfg=batch_shapes_cfg))

# =====================================================================
# OPTİMİZER — AMP + AdamW
# =====================================================================
optim_wrapper = dict(
    type='AmpOptimWrapper',  # FP16 mixed precision — VRAM tasarrufu + hız
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

# =====================================================================
# LR SCHEDULE — CosineAnnealing (by_epoch=True daha stabil)
# =====================================================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,      # 0.1→0.01: daha yavaş warmup
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=max_epochs,
        eta_min=base_lr * 0.01,  # Min LR = base_lr × 0.01
        begin=0,
        end=max_epochs)
]

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

# =====================================================================
# HOOKS
# =====================================================================
default_hooks = dict(
    checkpoint=dict(
        interval=5,            # 10→5: daha sık checkpoint
        max_keep_ckpts=5,      # 3→5: daha fazla checkpoint
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=500))  # 200→500

# =====================================================================
# LOSS WEIGHT TUNING
# =====================================================================
# 4 sınıflı küçük dataset için loss weight'ları ayarla
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
loss_dfl_weight = 1.5 / 4

# =====================================================================
# EVALUATOR
# =====================================================================
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = dict(ann_file=data_root + 'coco_annotations/instances_test.json')

# =====================================================================
# TRAIN CONFIG
# =====================================================================
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,            # 10→5: daha sık validation
    dynamic_intervals=[
        ((max_epochs - close_mosaic_epochs), 1)  # Son 20 epoch her epoch val
    ])

# =====================================================================
# CUSTOM HOOKS — PipelineSwitchHook epoch'u güncellendi
# =====================================================================
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

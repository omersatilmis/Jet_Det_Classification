# configs/cascade_rcnn_convnext_tiny.py
#
# Jet tespiti (F16, F18, F22, F35) için Cascade R-CNN ConvNeXt-Tiny-FPN config
# - MMDetection 3.x ile uyumlu
# - Windows için mp_start_method='spawn'
# - RCNN stage'lerinde pos_weight=-1 fix eklendi (AttributeError: pos_weight çözümü)
# - v2: CosineAnnealing LR, Mosaic/MixUp/CopyPaste augmentation, multi-scale
#   anchor, ClassBalanced thr artırıldı, TTA genişletildi, EMA optimize edildi

default_scope = 'mmdet'

# -------------------------------------------------------------------------
# 1. MODEL
# -------------------------------------------------------------------------
model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128-noema_in1k_20220221-5fa0f3ef.pth')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],  # Küçük nesneler için genişletilmiş scale
            ratios=[0.33, 0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),  # 10→2: cls/bbox dengesi
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1.0, 0.8, 0.6],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
            ),
        ],
    ),

    # ---------------- FIX HERE ----------------
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3,
                match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                    min_pos_iou=0.5, match_low_quality=False),
                sampler=dict(
                    type='RandomSampler', num=512,
                    pos_fraction=0.25, add_gt_as_proposals=True),
                pos_weight=-1,  
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner', pos_iou_thr=0.6, neg_iou_thr=0.6,
                    min_pos_iou=0.6),
                sampler=dict(
                    type='RandomSampler', num=512,
                    pos_fraction=0.25, add_gt_as_proposals=True),
                pos_weight=-1,  
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.7,
                    min_pos_iou=0.7),
                sampler=dict(
                    type='RandomSampler', num=512,
                    pos_fraction=0.25, add_gt_as_proposals=True),
                pos_weight=-1,
            ),
        ],
    ),

    test_cfg=dict(
        rpn=dict(
            nms_pre=1000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7)),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=50),  # 20→50: daha fazla tespit
    ),
)

# -------------------------------------------------------------------------
# 2. DATASET & DATALOADERS
# -------------------------------------------------------------------------

dataset_type = 'CocoDataset'
classes = ('F16', 'F18', 'F22', 'F35')

# Bu kısımlar train_jet_detection.py tarafından çalışma anında otomatik doldurulur.
# Varsayılan değerler projenin taşınabilirliğini bozmamak için boş bırakılmıştır.
data_root = ''
img_dir = '' # archive/ içindeki görseller için
ann_dir = ''

# ---------- Mosaic/MixUp için ayrı yardımcı pipeline ----------
# CachedMosaic ve CachedMixUp bellekte samples tutar (RAM-dostu).

train_pipeline_stage1 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(1333, 800),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 704), (1333, 768),
                (1333, 800), (1333, 864), (1333, 928),
                (1333, 960), (1333, 1024)],
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.9, 0.9),
        allow_negative_crop=False,
        recompute_bbox=True,
        bbox_clip_border=True),
    dict(
        type='CachedMixUp',
        img_scale=(1333, 800),
        ratio_range=(0.8, 1.6),
        max_cached_images=10,
        random_pop=True,
        pad_val=114.0),
    dict(type='CopyPaste', max_num_pasted=5),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32, filter_thr_px=1),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs'),
]

# Stage 2: Son epoch'larda Mosaic/MixUp kapatılır (fine-tune)
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 800), (1333, 960), (1333, 1024)],
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.9, 0.9),
        allow_negative_crop=False,
        recompute_bbox=True,
        bbox_clip_border=True),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=16, filter_thr_px=1),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs'),
]

# Varsayılan pipeline (eğitim stage1 ile başlar)
train_pipeline = train_pipeline_stage1

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1600, 1000), keep_ratio=True),  # 1333→1600: küçük nesne ↑
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.3,  # 0.1→0.3: F22 sınıfı daha fazla örneklenecek
        dataset=dict(
            type=dataset_type,
            metainfo=dict(classes=classes),
            data_root=data_root,
            ann_file='instances_train.json', # patch_cfg_paths() ile override edilir
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=16),
            pipeline=train_pipeline,
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='instances_validation.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='instances_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='instances_validation.json',
    metric='bbox',
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='instances_test.json',
    metric='bbox',
)

# -------------------------------------------------------------------------
# 3. OPTIMIZER & SCHEDULE
# -------------------------------------------------------------------------

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.0002, # ConvNeXt-Tiny için daha hizli baslangic
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=36,
        eta_min=1e-6,
        begin=0,
        end=36)  # MultiStepLR→CosineAnnealing: plato sorununu çözer
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=36,
    val_interval=1,
    dynamic_intervals=[(30, 1)],  # Son 6 epoch'ta her epoch val
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100),
)

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(1333, 640), keep_ratio=True),
                dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                dict(type='Resize', scale=(1333, 960), keep_ratio=True),
                dict(type='Resize', scale=(1600, 1200), keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=0.0),
                dict(type='RandomFlip', prob=1.0),
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [dict(type='PackDetInputs')],
        ])
]

# -------------------------------------------------------------------------
# 4. HOOKS & RUNTIME
# -------------------------------------------------------------------------

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# Windows FIX: fork -> spawn
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer',
)

log_level = 'INFO'
load_from = None
resume = False

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,  # 0.0002→0.0001: küçük batch ile daha smooth
        update_buffers=True),
    # Son 6 epoch'ta Mosaic/MixUp kapatılır (fine-tune aşaması)
    dict(
        type='PipelineSwitchHook',
        switch_epoch=30,
        switch_pipeline=train_pipeline_stage2)
]

# Eğitim & Değerlendirme Rehberi

F16, F18, F22 ve F35 jetlerini tespit eden iki model: **Cascade R-CNN** (MMDetection) ve **YOLOv8-S** (MMYOLO).

## 1. Ortam Kurulumu

Bkz. [SETUP_INSTALL.md](SETUP_INSTALL.md)

## 2. Veri Seti Hazırlığı

```bash
python shared/scripts/prepare_jet_dataset.py
```

Beklenen dosyalar:
- `archive/labels_with_split.csv`
- `archive/dataset/` (görseller)

Çıktılar: `coco_annotations/instances_{train,validation,test}.json`

## 3. Model Eğitimi

### Cascade R-CNN + ConvNeXt-Tiny (MMDetection)

```bash
# Sıfırdan eğitim
python mmdetection/scripts/train_jet_detection.py \
  --config mmdetection/configs/cascade_rcnn_convnext_tiny.py \
  --work-dir work_dirs/mmdetection/cascade_rcnn_convnext_tiny_v2 \
  --fresh

# Kaldığı yerden devam
python mmdetection/scripts/train_jet_detection.py \
  --config mmdetection/configs/cascade_rcnn_convnext_tiny.py \
  --work-dir work_dirs/mmdetection/cascade_rcnn_convnext_tiny_v2 \
  --resume
```

| Parametre | Değer |
|-----------|-------|
| Epochs | 36 |
| Batch size | 2 |
| LR | 0.0002 (AdamW + CosineAnnealing) |
| Augmentation | Mosaic + MixUp + CopyPaste |
| Tahmini süre | ~6-8 saat (RTX 2060) |

### YOLOv8-S (MMYOLO)

```bash
# Sıfırdan eğitim
python mmyolo/scripts/train_yolo_detection.py \
  --work-dir work_dirs/mmyolo/yolov8_s_jet_v2

# Kaldığı yerden devam
python mmyolo/scripts/train_yolo_detection.py \
  --work-dir work_dirs/mmyolo/yolov8_s_jet_v2 \
  --resume
```

| Parametre | Değer |
|-----------|-------|
| Epochs | 150 |
| Batch size | 4 |
| LR | 0.0008 (AdamW + CosineAnnealing) |
| Giriş boyutu | 800×800 |
| Augmentation | Mosaic + MixUp + Albumentations |
| Tahmini süre | ~3-4 saat (RTX 2060) |

### Multi-Seed Eğitim (Opsiyonel)
```bash
python mmdetection/scripts/train_jet_detection_multiseed.py
```

## 4. Değerlendirme

Her model için ayrı evaluator script'leri mevcuttur.

### Cascade R-CNN
```bash
python mmdetection/evaluation/evaluate_mmdet.py --split val
python mmdetection/evaluation/evaluate_mmdet.py --split test --tta
```

### YOLOv8-S
```bash
python mmyolo/evaluation/evaluate_mmyolo.py --split val
python mmyolo/evaluation/evaluate_mmyolo.py --split test
```

Her evaluator şu çıktıları üretir:
- `coco_metrics.json` — mAP, AP50, AP75, AP_s/m/l
- `confusion_matrix.png` + `.csv`
- `per_class_metrics.json` — Sınıf bazlı Precision/Recall/F1
- `error_examples.json` — FP/FN hatalı tespit listesi

## 5. Model Karşılaştırma

İki modelin sonuçlarını yan yana karşılaştırır:
```bash
python shared/scripts/compare_models.py --split val
```

Çıktılar (`outputs/comparison_val/`):
- `coco_comparison.png` — COCO metrikleri bar chart
- `per_class_f1_comparison.png` — Sınıf bazlı F1 karşılaştırması
- `comparison_report.json` — Detaylı rapor

## 6. FPS Benchmark

Her iki model için hız, parametre sayısı ve GPU memory ölçer:
```bash
python shared/scripts/benchmark_fps.py
python shared/scripts/benchmark_fps.py --num-images 200
python shared/scripts/benchmark_fps.py --only mmdet
```

Çıktılar (`outputs/benchmark/`):
- `benchmark_results.json` — Ham veriler
- `benchmark_table.md` — Teze kopyala-yapıştır tablo

## 7. Inference

### Video
```bash
python mmdetection/inference/video_test_infer.py \
  --video testing/video_input/sample.mp4 --save
```

### Notebook
- `mmdetection/inference/jet_infer_single_image_final.ipynb`
- `mmdetection/inference/jet_infer_test_class_image.ipynb`

## 8. RTX 2060 İçin Notlar

- Tüm config'ler AMP (FP16) kullanır — VRAM tasarrufu
- CUDA OOM olursa: `batch_size` değerini 1 yapın
- `num_workers=0` yaparak Windows donma sorunlarını önleyin
- TensorBoard ile eğitimi izleyin: `tensorboard --logdir work_dirs/`

## 9. Checkpoint Konumları

```
work_dirs/
├── mmdetection/
│   └── cascade_rcnn_convnext_tiny_v2/
│       ├── best_coco_bbox_mAP_epoch_X.pth   ← En iyi model
│       ├── epoch_*.pth                       ← Son 3-5 checkpoint
│       └── YYYYMMDD/vis_data/scalars.json    ← Eğitim logları
└── mmyolo/
    └── yolov8_s_jet_v2/
        ├── best_coco_bbox_mAP_epoch_X.pth
        ├── epoch_*.pth
        └── YYYYMMDD/vis_data/scalars.json
```

# Jet Uçak Tespiti ve Sınıflandırma

RTX 2060 (6 GB) için optimize edilmiş, **iki farklı model** ile jet uçağı tespit ve sınıflandırma projesi. (Bitirme Projesi)

## Özellikler

- **4 Sınıf**: F16, F18, F22, F35
- **Model 1**: Cascade R-CNN + ConvNeXt-Tiny-FPN (MMDetection 3.x)
- **Model 2**: YOLOv8-S (MMYOLO)
- **Veri Seti**: COCO formatında, 2,254 train / 532 val / 207 test görsel
- **Araçlar**: Eğitim, değerlendirme, FPS benchmark, model karşılaştırma, video inference, web UI

## Gereksinimler

### Donanım
- GPU: RTX 2060 (6 GB) veya üzeri
- RAM: 16 GB önerilir

### Yazılım
- Python 3.8+
- CUDA 11.6+ (PyTorch ile eşleşmeli)
- MMDetection 3.x, MMEngine, MMCV 2.x, MMYOLO

> Detaylı kurulum: [SETUP_INSTALL.md](SETUP_INSTALL.md)

## Hızlı Başlangıç

### 1. Veri Seti Hazırlığı
```bash
python shared/scripts/prepare_jet_dataset.py
```

### 2. Eğitim

**Cascade R-CNN (MMDetection):**
```bash
python mmdetection/scripts/train_jet_detection.py \
  --config mmdetection/configs/cascade_rcnn_convnext_tiny.py \
  --work-dir work_dirs/mmdetection/cascade_rcnn_convnext_tiny_v2 \
  --fresh
```

**YOLOv8-S (MMYOLO):**
```bash
python mmyolo/scripts/train_yolo_detection.py \
  --work-dir work_dirs/mmyolo/yolov8_s_jet_v2
```

### 3. Değerlendirme
```bash
python mmdetection/evaluation/evaluate_mmdet.py --split val
python mmyolo/evaluation/evaluate_mmyolo.py --split val
```

### 4. Model Karşılaştırma
```bash
python shared/scripts/compare_models.py --split val
```

### 5. FPS Benchmark
```bash
python shared/scripts/benchmark_fps.py
```

### 6. Inference
```bash
python mmdetection/inference/video_test_infer.py --video testing/video_input/sample.mp4 --save
```

> Detaylı eğitim rehberi: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## Model Konfigürasyonları

| Parametre | Cascade R-CNN | YOLOv8-S |
|-----------|--------------|----------|
| Backbone | ConvNeXt-Tiny | YOLOv8CSPDarknet |
| Neck | FPN | YOLOv8PAFPN |
| Giriş Boyutu | 1600×1000 | 800×800 |
| Optimizer | AdamW (lr=0.0002) | AdamW (lr=0.0008) |
| LR Schedule | CosineAnnealing | CosineAnnealing |
| Epochs | 36 | 150 |
| Batch Size | 2 | 4 |
| AMP (FP16) | ✅ | ✅ |
| Augmentation | Mosaic + MixUp + CopyPaste | Mosaic + MixUp + Albu |
| EMA | ✅ | ✅ |

## Dizin Yapısı

```
jet_detection_project/
├── mmdetection/
│   ├── configs/                # Cascade R-CNN config
│   ├── scripts/                # Eğitim scriptleri
│   ├── evaluation/             # MMDet model evaluator
│   └── inference/              # Inference araçları (notebook + video)
├── mmyolo/
│   ├── configs/jet/            # YOLOv8-S config
│   ├── scripts/                # MMYOLO eğitim scriptleri
│   └── evaluation/             # MMYOLO model evaluator
├── shared/scripts/
│   ├── prepare_jet_dataset.py  # CSV → COCO dönüşümü
│   ├── compare_models.py       # Model karşılaştırma aracı
│   └── benchmark_fps.py        # FPS & parametre benchmark
├── coco_annotations/           # COCO JSON dosyaları
├── dataset_analysis/           # Veri seti analiz grafikleri
├── outputs/                    # Eval/benchmark çıktıları
├── work_dirs/                  # Eğitim checkpoint'ları
│   ├── mmdetection/            # Cascade R-CNN checkpoint'ları
│   └── mmyolo/                 # YOLOv8 checkpoint'ları
├── testing/                    # Test verileri
├── AntJetDetUI/                # Web arayüzü (React + FastAPI)
├── SETUP_INSTALL.md            # Kurulum rehberi
├── TRAINING_GUIDE.md           # Eğitim rehberi
└── requirements.txt            # Python bağımlılıkları
```

## Sorun Giderme

- **CUDA OOM**: Config'de `batch_size=1` yapın veya görsel boyutunu küçültün
- **Yavaş eğitim**: AMP (`AmpOptimWrapper`) açık olduğundan emin olun
- **Eksik dosya**: `archive/dataset` ve `coco_annotations` yollarını kontrol edin
- **Checkpoint bulunamadı**: `work_dirs/` altında doğru klasörü kontrol edin

# Kurulum Rehberi — Jet Uçak Tespit Sistemi

MMDetection 3.x + MMYOLO + MMEngine + MMCV 2.x ortamı. RTX 2060 (6 GB) üzerinde test edilmiştir.

## 1. Sanal Ortam Oluşturun

```powershell
conda create -n jet_det python=3.10 -y
conda activate jet_det
```

## 2. PyTorch + CUDA

```powershell
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

> Farklı CUDA sürümü kullanıyorsanız https://pytorch.org/get-started/locally/ adresinden uygun komutu alın.

## 3. OpenMMLab Kütüphaneleri

```powershell
pip install mmengine==0.10.4
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install mmdet==3.3.0
pip install mmyolo==0.6.0
```

## 4. Diğer Bağımlılıklar

```powershell
pip install -r requirements.txt
```

Bu aşağıdakileri içerir:
- `opencv-python`, `pillow`, `numpy`, `pandas`
- `matplotlib`, `seaborn`, `tensorboard`
- `pycocotools`, `albumentations`
- `tqdm`

## 5. Kurulumu Test Edin

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()} ({torch.version.cuda})")
print(f"GPU: {torch.cuda.get_device_name(0)}")

from mmcv import __version__ as mmcv_v
from mmdet import __version__ as mmdet_v
from mmyolo import __version__ as mmyolo_v
from mmengine import __version__ as mmengine_v
print(f"mmcv={mmcv_v}, mmdet={mmdet_v}, mmyolo={mmyolo_v}, mmengine={mmengine_v}")

from mmdet.apis import init_detector
print("✅ Tüm import'lar başarılı!")
```

## Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `CUDA out of memory` | Config'de `batch_size=1` yapın |
| `mmcv bulunamıyor` | `pip install mmcv==2.1.0 -f ...` ile doğru CUDA/torch sürümünü kullanın |
| `torch.cuda.is_available() = False` | PyTorch'u CUDA destekli kurun, `nvidia-smi` ile GPU'yu kontrol edin |
| `ModuleNotFoundError: mmyolo` | `pip install mmyolo==0.6.0` |
| Windows'ta donma | Config'de `mp_start_method='spawn'` olduğunu doğrulayın |

## Notlar

- Kurulum sırası **önemlidir**: PyTorch → MMCV → MMEngine → MMDet → MMYOLO
- Windows'ta `mp_start_method='spawn'` kullanılmalıdır (config'lerde ayarlı)
- RTX 2060 için tüm config'ler AMP (FP16) kullanır — VRAM tasarrufu sağlar

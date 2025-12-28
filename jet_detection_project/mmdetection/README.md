# MMDetection Workspace

Bu klasor, projenin MMDetection tarafini duzenli ve moduler sekilde yonetmek icin olusturuldu.

## Dizinler

- `configs/`: MMDetection config dosyalari
- `datasets/`: `images/` ve `coco_annotations/` veri baglantilari
- `checkpoints/`: egitim agirliklari
- `work_dirs/`: MMEngine calisma dizinleri
- `scripts/`: veri hazirlama ve yardimci scriptler
- `tools/`: train/eval/inference giris scriptleri
- `inference/`: inference cikti ve yardimci notlar
- `evaluation/`: metrik raporlari
- `logs/`: run loglari
- `outputs/`: gorsel/json ciktilar
- `docs/`: plan ve teknik dokumanlar

## Hızlı Kullanım

```bash
python mmdetection/tools/train.py --config mmdetection/configs/cascade_rcnn_convnext_tiny.py --work-dir work_dirs/cascade_rcnn_convnext_tiny
python mmdetection/tools/eval.py --config mmdetection/configs/cascade_rcnn_convnext_tiny.py --work-dir work_dirs/cascade_rcnn_convnext_tiny
python mmdetection/tools/infer_video.py --video testing/video_input/sample.mp4 --save
```

Not: Wrapper scriptler `mmdetection/scripts` ve `mmdetection/inference` altindaki dosyalari cagirir.

Ortak veri hazirlama scriptleri `shared/scripts/` altindadir:
- `shared/scripts/prepare_jet_dataset.py`
- `shared/scripts/convert_csv_to_coco.py`
- `shared/scripts/analyze_coco_dataset.py`

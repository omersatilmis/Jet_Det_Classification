# MMDetection Klasor Yapisi

Bu yapi, hibrit sistemde MMDetection ve MMYOLO taraflarini birbirinden ayirmak icin tasarlandi.

- MMDetection odakli gelistirmeler: `mmdetection/`
- MMYOLO odakli gelistirmeler: `mmyolo/`
- Ortak (framework bagimsiz) scriptler: `shared/scripts/`

MMYOLO tarafinda ortak script cagri wrapperlari:
- `mmyolo/scripts/prepare_shared_dataset.py`
- `mmyolo/scripts/convert_shared_csv_to_coco.py`
- `mmyolo/scripts/analyze_shared_coco.py`
- `mmyolo/scripts/train_yolo_detection.py`

MMYOLO proje-ozel config:
- `mmyolo/configs/jet/yolov8_s_jet.py`

Bir sonraki adimda `configs/` altina MMDetection icin proje-ozel config kopyalari alinabilir.

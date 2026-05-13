from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

# 1. Create a minimal GT file with only Image 1
ann_file = "c:/Users/omerf/Desktop/Jet_Sentinel_Project/jet_detection_project/coco_annotations/instances_test.json"
coco_full = COCO(ann_file)
img_id = 1
img_info = coco_full.loadImgs([img_id])[0]
ann_ids = coco_full.getAnnIds(imgIds=[img_id])
anns = coco_full.loadAnns(ann_ids)

subset_gt = {
    "images": [img_info],
    "annotations": anns,
    "categories": coco_full.loadCats(coco_full.getCatIds())
}

subset_ann_path = "subset_gt.json"
with open(subset_ann_path, 'w') as f:
    json.dump(subset_gt, f)

# 2. Create a prediction file for Image 1 (using the values from log)
# DEBUG Prediction: img=1, cat=36 (F35), bbox=[1646.5, 1104.9, 119.6, 36.1], score=0.851
prediction = [
    {
        "image_id": 1,
        "category_id": 36,
        "bbox": [1646.5, 1104.9, 119.6, 36.1],
        "score": 0.851
    }
]
pred_path = "subset_pred.json"
with open(pred_path, 'w') as f:
    json.dump(prediction, f)

# 3. Evaluate
print("\n--- SINGLE IMAGE EVALUATION (Image ID 1) ---")
coco_gt = COCO(subset_ann_path)
coco_dt = coco_gt.loadRes(pred_path)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.params.imgIds = [1]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

os.remove(subset_ann_path)
os.remove(pred_path)

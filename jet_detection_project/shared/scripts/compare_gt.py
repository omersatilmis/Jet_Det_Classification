from pycocotools.coco import COCO
import json

ann_file = "c:/Users/omerf/Desktop/Jet_Sentinel_Project/jet_detection_project/coco_annotations/instances_test.json"
coco = COCO(ann_file)

img_id = 1
img_info = coco.loadImgs([img_id])[0]
print(f"Image ID {img_id}: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")

ann_ids = coco.getAnnIds(imgIds=[img_id])
anns = coco.loadAnns(ann_ids)

print("\nGround Truth Annotations for Image 1:")
for ann in anns:
    cat = coco.loadCats([ann['category_id']])[0]
    print(f"  Category: {ann['category_id']} ({cat['name']}), BBox: {ann['bbox']}")

print("\nMy Previous Log Prediction for Image 1:")
# img=1, cat=36 (F35), bbox=[1646.5, 1104.9, 119.6, 36.1], score=0.851
print("  Category: 36 (F35), BBox: [1646.5, 1104.9, 119.6, 36.1]")

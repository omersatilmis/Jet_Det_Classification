from pycocotools.coco import COCO
import os

ann_file = "c:/Users/omerf/Desktop/Jet_Sentinel_Project/jet_detection_project/coco_annotations/instances_test.json"
coco = COCO(ann_file)

target_classes = ['F16', 'F18', 'F22', 'F35']
cat_ids = coco.getCatIds(catNms=target_classes)
print(f"Target Category IDs: {cat_ids}")

img_ids = []
for cid in cat_ids:
    ids = coco.getImgIds(catIds=[cid])
    img_ids.extend(ids)

img_ids = sorted(list(set(img_ids)))
print(f"Found {len(img_ids)} total images containing target jets.")
print(f"First 20 valid image IDs: {img_ids[:20]}")

# Also check one image filename to be sure about paths
if img_ids:
    img_info = coco.loadImgs([img_ids[0]])[0]
    print(f"Sample filename in JSON: {img_info['file_name']}")

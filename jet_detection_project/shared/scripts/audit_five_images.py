from pycocotools.coco import COCO
import os
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# Setup
project_root = "c:/Users/omerf/Desktop/Jet_Sentinel_Project/jet_detection_project"
ann_file = os.path.join(project_root, "coco_annotations/instances_test.json")
config_file = os.path.join(project_root, "work_dirs/cascade_rcnn_r50_tiny/cascade_rcnn_r50_tiny.py")
checkpoint_file = os.path.join(project_root, "work_dirs/cascade_rcnn_r50_tiny/best_coco_bbox_mAP_epoch_21.pth")
archive_dir = "D:/Jet-Projesi/archive/dataset"

register_all_modules()
model = init_detector(config_file, checkpoint_file, device='cuda:0')
coco = COCO(ann_file)

target_classes = ['F16', 'F18', 'F22', 'F35']
cat_ids = coco.getCatIds(catNms=target_classes)
jet_img_ids = sorted(list(set(sum([coco.getImgIds(catIds=[cid]) for cid in cat_ids], []))))[:5]

print(f"Auditing Jet Image IDs: {jet_img_ids}")

for img_id in jet_img_ids:
    img_info = coco.loadImgs([img_id])[0]
    img_path = os.path.join(archive_dir, img_info['file_name'])
    
    print(f"\n--- IMAGE {img_id}: {img_info['file_name']} ---")
    if not os.path.exists(img_path):
        print(f"  [ERROR] Image not found at {img_path}")
        continue
    
    # Ground Truth
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    print("  GROUND TRUTH:")
    for ann in anns:
        cat_name = coco.loadCats([ann['category_id']])[0]['name']
        print(f"    - Cat: {ann['category_id']} ({cat_name}), BBox: {ann['bbox']}")
    
    # Prediction
    result = inference_detector(model, img_path)
    print("  PREDICTIONS (Score > 0.3):")
    if hasattr(result, 'pred_instances'):
        instances = result.pred_instances
        bboxes = instances.bboxes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        labels = instances.labels.cpu().numpy()
        classes = model.dataset_meta['classes']
        
        found = False
        for b, s, l in zip(bboxes, scores, labels):
            if s < 0.3: continue
            found = True
            c_name = classes[l]
            print(f"    - Cat: {c_name}, Score: {s:.3f}, BBox: [{b[0]:.1f}, {b[1]:.1f}, {b[2]-b[0]:.1f}, {b[3]-b[1]:.1f}]")
        if not found:
             print("    - (None above 0.3)")

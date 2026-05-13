from pycocotools.coco import COCO

ann_file = "c:/Users/omerf/Desktop/Jet_Sentinel_Project/jet_detection_project/coco_annotations/instances_test.json"
coco = COCO(ann_file)

all_img_ids = coco.getImgIds()
target_classes = ['F16', 'F18', 'F22', 'F35']
cat_ids = coco.getCatIds(catNms=target_classes)

total_jet_anns = 0
imgs_with_jets = set()
for cid in cat_ids:
    ann_ids = coco.getAnnIds(catIds=[cid])
    total_jet_anns += len(ann_ids)
    imgs_with_jets.update(coco.getImgIds(catIds=[cid]))

print(f"Total images in JSON: {len(all_img_ids)}")
print(f"Total jet annotations (F16, F18, F22, F35): {total_jet_anns}")
print(f"Total images containing at least one target jet: {len(imgs_with_jets)}")

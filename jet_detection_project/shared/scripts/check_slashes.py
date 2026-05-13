import json

ann_file = "c:/Users/omerf/Desktop/Jet_Sentinel_Project/jet_detection_project/coco_annotations/instances_test.json"
with open(ann_file, 'r') as f:
    data = json.load(f)

filenames = [img['file_name'] for img in data['images']]
total = len(filenames)
with_slash = [f for f in filenames if '/' in f or '\\' in f]

print(f"Total images in JSON: {total}")
print(f"Images with directory slashes: {len(with_slash)}")
if with_slash:
    print(f"Sample with slash: {with_slash[0]}")
else:
    print("No images have directory slashes in the JSON.")

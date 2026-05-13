import json

ann_file = "c:/Users/omerf/Desktop/Jet_Sentinel_Project/jet_detection_project/coco_annotations/instances_test.json"
print(f"Loading {ann_file}...")
with open(ann_file, 'r') as f:
    data = json.load(f)

print("\nAll Categories in JSON:")
for cat in data['categories']:
    print(f"  {cat['id']}: {cat['name']}")

# List all just in case
# print(json.dumps(data['categories'], indent=2))

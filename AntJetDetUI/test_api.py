"""Test the /api/analyze-single endpoint with a real JPEG image."""
import cv2
import numpy as np
import requests
import time

# Create a real test image (640x640 black image)
img = np.zeros((640, 640, 3), dtype=np.uint8)
cv2.imwrite('test_image.jpg', img)

print('Sending inference request to http://127.0.0.1:8000/api/analyze-single ...')
start = time.time()

with open('test_image.jpg', 'rb') as f:
    resp = requests.post(
        'http://127.0.0.1:8000/api/analyze-single',
        data={'model_id': 'cascade-rcnn'},
        files={'file': ('test_image.jpg', f, 'image/jpeg')},
        timeout=120
    )

elapsed = time.time() - start
print(f'Status Code: {resp.status_code}  (took {elapsed:.1f}s)')

if resp.status_code == 200:
    data = resp.json()
    print(f'  status: {data["status"]}')
    print(f'  models: {len(data["models"])}')
    if data["models"]:
        m = data["models"][0]
        print(f'  model_name: {m["model_name"]}')
        print(f'  detections: {len(m["detections"])}')
        print(f'  inference_time_ms: {m["metrics"]["inference_time_ms"]:.1f}')
        print(f'  has visualized_image: {bool(m.get("visualized_image"))}')
        print(f'  has heatmap_image: {bool(m.get("heatmap_image"))}')
    print('\n*** SUCCESS! API is working. ***')
else:
    print(f'FAILED! Response: {resp.text[:500]}')

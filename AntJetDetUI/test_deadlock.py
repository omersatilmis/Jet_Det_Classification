import sys
from backend.inference import init_cascade_rcnn_r50
import numpy as np

print('Initializing model...')
model = init_cascade_rcnn_r50()
print('Model initialized. Creating test image...')
img = np.zeros((640, 640, 3), dtype=np.uint8)

print('Running raw inference_detector...')
from mmdet.apis import inference_detector
result = inference_detector(model, img)
print('SUCCESS! Inference completed without deadlock.')

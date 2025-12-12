import cv2
import base64
import numpy as np

def generate_visualizer_image(model, img: np.ndarray, result, confidence_thr: float) -> str:
    """Uses MMDetection's native visualizer to draw labeled bounding boxes, encoding to Base64.
    
    This feeds the Output Tab (the final resulting image).
    """
    from mmdet.registry import VISUALIZERS
    
    visualizer = VISUALIZERS.build(dict(type='DetLocalVisualizer', name='vis', line_width=3, alpha=0.8))
    visualizer.dataset_meta = model.dataset_meta
    
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    visualizer.add_datasample('pred', frame_rgb, data_sample=result, draw_gt=False, show=False, pred_score_thr=confidence_thr)
    
    out_bgr = cv2.cvtColor(visualizer.get_image(), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', out_bgr)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

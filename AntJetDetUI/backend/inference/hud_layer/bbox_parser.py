from typing import Dict, Any, List

def parse_detections(result, img_shape, confidence_thr: float, class_names: dict) -> List[Dict[str, Any]]:
    """Filters inference results by confidence and normalizes bounding box coordinates (0-1 range).
    
    This feeds data primarily to the HUD Layer and Objects Tab.
    """
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    
    img_h, img_w, _ = img_shape
    detections = []
    
    for i in range(len(scores)):
        if scores[i] >= confidence_thr:
            x1, y1, x2, y2 = bboxes[i]
            detections.append({
                "class_name": class_names.get(int(labels[i]), f"Unknown-{labels[i]}"),
                "confidence": float(scores[i]),
                "box": {
                    "x": float(x1 / img_w),
                    "y": float(y1 / img_h),
                    "width": float((x2 - x1) / img_w),
                    "height": float((y2 - y1) / img_h)
                }
            })
    return detections

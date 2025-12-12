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
    
    import math # For distance calculation
    
    for i in range(len(scores)):
        if scores[i] >= confidence_thr:
            x1, y1, x2, y2 = bboxes[i]
            
            # Normalized coordinates (0-1)
            nx = float(x1 / img_w)
            ny = float(y1 / img_h)
            nw = float((x2 - x1) / img_w)
            nh = float((y2 - y1) / img_h)
            
            # 1. Azimuth & Elevation (Relative to center)
            center_x = nx + (nw / 2)
            center_y = ny + (nh / 2)
            
            # Assuming ~60deg Horizontal FOV / ~40deg Vertical FOV
            azimuth = (center_x - 0.5) * 60.0
            elevation = (0.5 - center_y) * 40.0 # Standard: positive is UP
            
            # 2. Distance Estimation (Inverse square law approximation)
            # Reference: A jet occupying 10% of image height is approx 1.5km away
            area = nw * nh
            if area > 0:
                distance_km = 0.15 / math.sqrt(area)
            else:
                distance_km = 0.0

            detections.append({
                "class_name": class_names.get(int(labels[i]), f"Unknown-{labels[i]}"),
                "confidence": float(scores[i]),
                "box": {
                    "x": float(nx),
                    "y": float(ny),
                    "width": float(nw),
                    "height": float(nh)
                },
                "azimuth": float(round(azimuth, 2)),
                "elevation": float(round(elevation, 2)),
                "distance_km": float(round(distance_km, 2))
            })
    return detections

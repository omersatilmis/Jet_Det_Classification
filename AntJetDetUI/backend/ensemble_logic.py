import numpy as np
from typing import List, Dict, Any

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2
    
    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
        
    return inter_area / union_area

def weighted_box_fusion(detections_by_model: List[List[Dict[str, Any]]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Simplified Weighted Box Fusion.
    Groups boxes from different models that represent the same object.
    """
    if not detections_by_model:
        return []
        
    # Flatten all detections into a single list with model index
    all_detections = []
    for model_idx, detections in enumerate(detections_by_model):
        for d in detections:
            # Create a flat box list for IoU calculation: [x, y, w, h]
            box = d.get('box', {})
            box_list = [box.get('x', 0), box.get('y', 0), box.get('width', 0), box.get('height', 0)]
            all_detections.append({**d, 'model_idx': model_idx, 'box_list': box_list})
            
    if not all_detections:
        return []
        
    # Sort by confidence descending
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    clusters = []
    for det in all_detections:
        matched = False
        for cluster in clusters:
            # Compare with the first (highest confidence) box in the cluster
            det_class = det.get("class_name")
            cluster_class = cluster[0].get("class_name")
            if det_class and cluster_class and det_class == cluster_class and calculate_iou(det['box_list'], cluster[0]['box_list']) > iou_threshold:
                cluster.append(det)
                matched = True
                break
        if not matched:
            clusters.append([det])
            
    fused_results = []
    num_models = len(detections_by_model)
    
    for cluster in clusters:
        # Weighted average of coordinates
        total_conf = sum(d['confidence'] for d in cluster)
        avg_x = sum(d['box_list'][0] * d['confidence'] for d in cluster) / total_conf
        avg_y = sum(d['box_list'][1] * d['confidence'] for d in cluster) / total_conf
        avg_w = sum(d['box_list'][2] * d['confidence'] for d in cluster) / total_conf
        avg_h = sum(d['box_list'][3] * d['confidence'] for d in cluster) / total_conf
        
        avg_conf = total_conf / len(cluster)
        
        # Agreement score (how many models detected this)
        contributing_models = len(set(d['model_idx'] for d in cluster))
        agreement = contributing_models / num_models
        
        fused_results.append({
            "class_name": cluster[0]["class_name"],
            "confidence": avg_conf,
            "box": {"x": avg_x, "y": avg_y, "width": avg_w, "height": avg_h},
            "agreement": agreement,
            "model_count": contributing_models
        })
        
    return fused_results

def calculate_ensemble_metrics(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates average metrics across all model runs.
    """
    if not model_results:
        return {
            "avg_inference_time_ms": 0,
            "avg_gpu_usage": 0,
            "avg_vram_usage_mb": 0,
            "avg_fps": 0,
            "avg_map": None,
            "avg_iou": None,
            "consensus_score": 0
        }
        
    total_time = sum(m["metrics"]["inference_time_ms"] for m in model_results)
    total_gpu = sum(m["metrics"].get("gpu_usage", 0) for m in model_results)
    total_vram = sum(m["metrics"].get("vram_usage_mb", 0) for m in model_results)
    total_fps = sum(m["metrics"].get("fps", 0) for m in model_results)
    map_values = [m["metrics"].get("map") for m in model_results if m["metrics"].get("map") is not None]
    iou_values = [m["metrics"].get("iou") for m in model_results if m["metrics"].get("iou") is not None]
    
    return {
        "avg_inference_time_ms": total_time / len(model_results),
        "avg_gpu_usage": total_gpu / len(model_results),
        "avg_vram_usage_mb": total_vram / len(model_results),
        "avg_fps": total_fps / len(model_results),
        "avg_map": (sum(map_values) / len(map_values)) if map_values else None,
        "avg_iou": (sum(iou_values) / len(iou_values)) if iou_values else None,
        "consensus_score": 0.0 # Will be updated after WBF if needed
    }

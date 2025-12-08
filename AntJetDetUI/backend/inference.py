import os
import torch
import numpy as np
import GPUtil
import psutil
import time
from typing import Dict, Any, List
from pydantic import BaseModel

# Initialize these as globals so we only load them once
global_model = None
global_class_names = {
    0: "F-16 Fighting Falcon",
    1: "F/A-18 Hornet",
    2: "F-22 Raptor",
    3: "F-35 Lightning II"
}

# Global model cache to prevent reloading
global_models = {}

def init_cascade_rcnn_r50():
    """
    Initializes the hardcoded MMDetection Cascade R-CNN R50 model.
    """
    global global_model
    if global_model is not None:
        return global_model

    try:
        from mmdet.apis import init_detector
        
        # Absolute paths based on the jet_detection_project location
        workspace_dir = r"C:\Users\omerf\Desktop\Jet_Det_Project\jet_detection_project"
        config_file = os.path.join(workspace_dir, "work_dirs", "cascade_rcnn_r50_tiny", "cascade_rcnn_r50_tiny.py")
        checkpoint_file = os.path.join(workspace_dir, "work_dirs", "cascade_rcnn_r50_tiny", "best_coco_bbox_mAP_epoch_21.pth")
        
        if not os.path.exists(config_file):
            print(f"[ERROR] Missing config file: {config_file}")
            return None
        if not os.path.exists(checkpoint_file):
            print(f"[ERROR] Missing checkpoint file: {checkpoint_file}")
            return None

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Add the workspace to sys.path to resolve any custom modules
        import sys
        if workspace_dir not in sys.path:
            sys.path.insert(0, workspace_dir)

        print(f"[INFO] Loading Cascade R-CNN R50 model onto {device}...")
        model = init_detector(config_file, checkpoint_file, device=device)
        print("[INFO] Model loaded successfully.")
        
        global_model = model
        return model
    except Exception as e:
        print(f"[ERROR] Failed to initialize MMDetection model: {e}")
        return None

def init_custom_model(model_info: Dict[str, Any]):
    """
    Initializes a custom model uploaded by the user.
    Maps the selected architecture to the correct predefined .py config.
    """
    model_id = model_info["id"]
    global global_models
    if model_id in global_models:
        return global_models[model_id]

    try:
        from mmdet.apis import init_detector
        
        workspace_dir = r"C:\Users\omerf\Desktop\Jet_Det_Project\jet_detection_project"
        import sys
        if workspace_dir not in sys.path:
            sys.path.insert(0, workspace_dir)
            
        try:
            import mmyolo
        except ImportError:
            pass
            
        # Map front-end architecture strings to local config files
        config_map = {
            "cascade_rcnn_r50_tiny": os.path.join(workspace_dir, "work_dirs", "cascade_rcnn_r50_tiny", "cascade_rcnn_r50_tiny.py"),
            "cascade_rcnn_convnext_tiny": os.path.join(workspace_dir, "mmdetection", "configs", "cascade_rcnn_convnext_tiny.py"),
            "yolov8_s_jet": os.path.join(workspace_dir, "mmyolo", "configs", "jet", "yolov8_s_jet.py")
        }
        
        architecture = model_info.get("architecture", "cascade_rcnn_r50_tiny")
        config_file = config_map.get(architecture)
        checkpoint_file = model_info.get("file_path")
        
        if not config_file or not os.path.exists(config_file):
            print(f"[ERROR] Custom init failed. Missing config: {config_file}")
            return None
            
        if not checkpoint_file or not os.path.exists(checkpoint_file):
            print(f"[ERROR] Custom init failed. Missing .pth file: {checkpoint_file}")
            return None

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        print(f"[INFO] Loading Custom Model '{model_info['name']}' ({architecture}) onto {device}...")
        model = init_detector(config_file, checkpoint_file, device=device)
        print("[INFO] Custom Model loaded successfully.")
        
        global_models[model_id] = model
        return model
    except Exception as e:
        print(f"[ERROR] Failed to initialize custom model: {e}")
        return None

from inference_lib.performance.metrics_extractor import get_hardware_metrics
from inference_lib.hud_layer.bbox_parser import parse_detections
from inference_lib.output.visualizer import generate_visualizer_image
from inference_lib.academic.xai_heatmap import generate_xai_heatmap

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decodes raw bytes into an OpenCV BGR image tensor."""
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def _run_inference_with_cam(model, img: np.ndarray):
    """
    Executes the MMDetection forward pass.
    Intercepts the Feature Pyramid Network (FPN) layer to capture deep activation maps.
    Returns the detection results and the captured activation tensors.
    """
    from mmdet.apis import inference_detector
    
    activation_maps = []
    def hook_fn(module, input, output):
        activation_maps.append(output.cpu().detach())
        
    handle = None
    if hasattr(model, 'neck') and hasattr(model.neck, 'fpn_convs'):
        handle = model.neck.fpn_convs[0].conv.register_forward_hook(hook_fn)
        
    result = inference_detector(model, img)
    
    if handle is not None:
        handle.remove()
        
    return result, activation_maps


def run_inference(image_bytes: bytes, model_id: str = "cascade-rcnn", custom_models_registry: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main Orchestrator for analyzing a single image.
    Delegates decoding, inference, and formatting to appropriate clean architecture modules.
    """
    start_time = time.time()
    
    # 1. Resolve Model
    model = None
    if model_id == "cascade-rcnn":
        model = init_cascade_rcnn_r50()
    elif custom_models_registry and model_id in custom_models_registry:
        model = init_custom_model(custom_models_registry[model_id])
    else:
        print(f"[ERROR] Request made for unknown model_id: {model_id}")
        
    if model is None:
        return {
            "success": False,
            "error": "Model initialization failed. Weights or dependencies are missing.",
            "metrics": {"inference_time_ms": 0, "fps": 0, "gpu_usage": 0, "vram_usage_mb": 0}
        }
    
    try:
        CONFIDENCE_THRESHOLD = 0.3
        
        # 1. Decode Image Layer
        img = _decode_image(image_bytes)
        
        # 2. XAI + Deep Learning Inference Layer
        result, activation_maps = _run_inference_with_cam(model, img)
        
        # 3. Data Parsing Layer (Feeds HUD and Objects Tabs)
        detections = parse_detections(result, img.shape, CONFIDENCE_THRESHOLD, global_class_names)
        
        # 4. Presentation / Visuals Layer (Feeds Output and Academic Tabs)
        visualized_string = generate_visualizer_image(model, img, result, CONFIDENCE_THRESHOLD)
        heatmap_string = generate_xai_heatmap(activation_maps, img)
        
        # 5. Telemetry Layer (Feeds Performance Tab)
        inference_time_ms = float((time.time() - start_time) * 1000)
        fps = float(1000.0 / inference_time_ms) if inference_time_ms > 0 else 0.0
        
        return {
            "success": True,
            "detections": detections,
            "visualized_image": visualized_string,
            "heatmap_image": heatmap_string,
            "metrics": {
                "inference_time_ms": inference_time_ms,
                "fps": fps,
                **get_hardware_metrics()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Inference orchestration failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "metrics": {"inference_time_ms": 0, "fps": 0, "gpu_usage": 0, "vram_usage_mb": 0}
        }

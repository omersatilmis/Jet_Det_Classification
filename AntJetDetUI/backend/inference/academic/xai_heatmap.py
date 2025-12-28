import numpy as np
import torch
import cv2
import base64

def generate_xai_heatmap(activation_maps: list, img: np.ndarray) -> str:
    """Generates a Grad-CAM style Explainable AI heatmap from deep activation tensors, encoding to Base64.
    
    This directly feeds the Academic Tab visualization.
    """
    if len(activation_maps) == 0:
        _, buffer = cv2.imencode('.jpg', img)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
    img_h, img_w, _ = img.shape
    fmap = activation_maps[0][0] # Focus on the first Batch
    
    # Average across all semantic channels to get holistic spatial attention
    spatial_attention = torch.mean(fmap, dim=0).numpy()
    
    # Normalize 0-1
    spatial_attention = np.maximum(spatial_attention, 0)
    if np.max(spatial_attention) > 0:
        spatial_attention /= np.max(spatial_attention)
        
    spatial_attention = cv2.resize(spatial_attention, (img_w, img_h))
    
    # Color mapping and overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * spatial_attention), cv2.COLORMAP_JET)
    gray_img_color = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    
    mask = spatial_attention > 0.20 # Ignore subtle low-confidence background noise
    superimposed_img = gray_img_color.copy()
    superimposed_img[mask] = cv2.addWeighted(gray_img_color[mask], 0.4, heatmap_color[mask], 0.6, 0)
    
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

import base64
import logging
from typing import List, Optional

import cv2
import numpy as np
import torch

# Loglama ayarları (Uygulamanın genel log ayarlarına entegre olur)
logger = logging.getLogger(__name__)

def _encode_image_to_base64(img: np.ndarray, ext: str = '.jpg') -> str:
    """
    Görüntüyü belirtilen formatta Base64 string'e dönüştürür.
    """
    success, buffer = cv2.imencode(ext, img)
    if not success:
        logger.error(f"Görüntü {ext} formatına dönüştürülemedi.")
        raise ValueError("Görüntü encode edilemedi.")
    
    # MIME tipini dinamik olarak ayarlayalım
    mime_type = "jpeg" if ext.lower() in [".jpg", ".jpeg"] else ext.lower().strip('.')
    return f"data:image/{mime_type};base64,{base64.b64encode(buffer).decode('utf-8')}"


def generate_xai_heatmap(
    activation_maps: List[torch.Tensor],
    img: np.ndarray,
    threshold: float = 0.40,
    overlay_alpha: float = 0.3,
    overlay_beta: float = 0.7,
    colormap: int = cv2.COLORMAP_JET
) -> str:
    """
    Generates a Grad-CAM style Explainable AI heatmap from deep activation tensors, 
    encoding the result to a Base64 string.
    
    Args:
        activation_maps: List of activation tensors hooked from CNN/Transformer layers.
        img: Original input image as a NumPy array (H, W, C).
        threshold: Activation threshold (0.0 to 1.0) to filter background noise.
        overlay_alpha: Weight of the original grayscale image in the overlay.
        overlay_beta: Weight of the heatmap in the overlay.
        colormap: OpenCV colormap to use for the heatmap.
        
    Returns:
        Base64 encoded string of the final heatmap image.
    """
    # Guard clause: Aktivasyon haritası yoksa orijinal görüntüyü döndür
    if not activation_maps or len(activation_maps) == 0:
        logger.warning("Aktivasyon haritası bulunamadı. Orijinal görüntü döndürülüyor.")
        return _encode_image_to_base64(img)

    try:
        img_h, img_w = img.shape[:2]
        
        # GPU'da olma veya Gradient takibi ihtimaline karşı güvenli çıkarma
        feature_map = activation_maps[0][0].detach().cpu()
        
        # Kanal bazında ortalama alma [H, W]
        spatial_attention = torch.mean(feature_map, dim=0).numpy()
        
        # ReLU benzeri kırpma: Negatif değerleri yoksay
        spatial_attention = np.maximum(spatial_attention, 0)
        
        # Dinamik bulanıklaştırma kernel'i hesaplama
        blur_kernel = max(3, spatial_attention.shape[0] // 10)
        blur_kernel = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        
        spatial_attention = cv2.GaussianBlur(spatial_attention, (blur_kernel, blur_kernel), 0)

        # Aykırı değer (outlier) etkisini azaltmak için yüzdelik normalizasyon
        p_high = np.percentile(spatial_attention, 98)
        p_low = np.percentile(spatial_attention, 2)
        
        if p_high > p_low:
            spatial_attention = (spatial_attention - p_low) / (p_high - p_low)
        
        spatial_attention = np.clip(spatial_attention, 0.0, 1.0)
        
        # Orijinal görüntü boyutlarına büyütme (Upsampling)
        spatial_attention = cv2.resize(spatial_attention, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        
        # Renk haritası oluşturma
        heatmap_color = cv2.applyColorMap(np.uint8(255 * spatial_attention), colormap)
        
        # Orijinal görüntüyü arka plan için gri tonlamaya çevirip 3 kanallı yapma
        gray_img_color = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        
        # Maskeleme ve Bindirme (Overlay)
        mask = spatial_attention > threshold
        final_img = gray_img_color.copy()
        
        final_img[mask] = cv2.addWeighted(
            gray_img_color[mask], overlay_alpha, 
            heatmap_color[mask], overlay_beta, 
            0
        )

        return _encode_image_to_base64(final_img)

    except Exception as e:
        logger.error(f"XAI Heatmap oluşturulurken beklenmeyen bir hata oluştu: {str(e)}", exc_info=True)
        # Hata durumunda sistemin çökmemesi için ham resmi geri döndürmek genellikle iyi bir fail-safe pratiğidir
        return _encode_image_to_base64(img)
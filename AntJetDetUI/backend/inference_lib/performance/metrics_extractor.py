import psutil
import GPUtil
from typing import Dict

def get_hardware_metrics() -> Dict[str, float]:
    """Reads actual hardware metrics from the server/PC."""
    gpu_usage = 0.0
    vram_usage = 0.0
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            # Get VRAM usage of the first GPU in MB
            gpu_usage = float(gpus[0].load * 100.0)
            vram_usage = float(gpus[0].memoryUsed)
    except Exception:
        pass
        
    return {
        "gpu_usage": float(gpu_usage),
        "vram_usage_mb": float(vram_usage)
    }

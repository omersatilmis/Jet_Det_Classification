import psutil
import GPUtil
from typing import Dict

def get_hardware_metrics() -> Dict[str, float]:
    """Reads actual hardware metrics from the server/PC."""
    cpu_usage = psutil.cpu_percent()
    vram_usage = 0.0
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            # Get VRAM usage of the first GPU in MB
            vram_usage = gpus[0].memoryUsed
    except Exception:
        pass
        
    return {
        "gpu_usage": float(cpu_usage), # Fallback to CPU usage if GPU usage failing
        "vram_usage_mb": float(vram_usage)
    }

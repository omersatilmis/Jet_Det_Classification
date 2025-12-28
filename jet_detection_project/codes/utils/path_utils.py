import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Proje kök dizinini (jet_detection_project) bulur.
    Codes klasörünün bir üst dizinidir.
    """
    return Path(__file__).resolve().parent.parent.parent

def get_data_paths():
    """
    Veri seti ve anotasyon yollarını merkezi olarak döndürür.
    """
    root = get_project_root()
    return {
        "project_root": root,
        "images": root.parent / "archive" / "dataset",
        "ann_dir": root / "coco_annotations",
        "work_dirs": root / "work_dirs"
    }

def ensure_dirs():
    """Gerekli klasörlerin varlığını kontrol eder."""
    paths = get_data_paths()
    for name, p in paths.items():
        if not p.exists() and name != "work_dirs":
            print(f"[WARNING] {name} yolu bulunamadı: {p}")
        elif name == "work_dirs":
            p.mkdir(parents=True, exist_ok=True)

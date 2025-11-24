import sys
import shutil
import subprocess
from pathlib import Path

CLASSES = ["F16", "F18", "F22", "F35"]

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = REPO_ROOT / "shared"
ARCHIVE_DIR = REPO_ROOT / "archive"

CSV_PATH = ARCHIVE_DIR / "labels_with_split.csv"
IMAGES_DIR = ARCHIVE_DIR / "dataset"

CONVERT_SCRIPT = SHARED_ROOT / "scripts" / "convert_csv_to_coco.py"
OUTPUT_DIR = REPO_ROOT / "coco_annotations"


def must_exist(p: Path, desc: str):
    if not p.exists():
        raise FileNotFoundError(f"{desc} bulunamadı: {p}")


def main():
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("usage: python shared/scripts/prepare_jet_dataset.py")
        print()
        print("Prepare jet dataset from CSV and produce COCO annotations.")
        print("Expected inputs:")
        print("  - archive/labels_with_split.csv")
        print("  - archive/dataset/")
        print("Output:")
        print("  - coco_annotations/instances_{train,validation,test}.json")
        return

    print("[INFO] Dataset hazırlama başlıyor...")

    must_exist(CSV_PATH, "labels_with_split.csv")
    must_exist(IMAGES_DIR, "archive/dataset klasörü")
    must_exist(CONVERT_SCRIPT, "convert_csv_to_coco.py")

    if OUTPUT_DIR.exists():
        print(f"[INFO] Var olan output siliniyor: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # convert_csv_to_coco.py argümanlarıyla UYUMLU çağrı:
    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        "--csv", str(CSV_PATH),
        "--images-dir", str(IMAGES_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--classes", *CLASSES
    ]

    print("[INFO] COCO dönüşümü çalışıyor:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("[OK] COCO anotasyonları üretildi:")
    print(f" - {OUTPUT_DIR / 'instances_train.json'}")
    print(f" - {OUTPUT_DIR / 'instances_validation.json'}")
    print(f" - {OUTPUT_DIR / 'instances_test.json'}")


if __name__ == "__main__":
    main()

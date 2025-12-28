import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    target = project_root / "mmdetection" / "scripts" / "train_jet_detection.py"
    cmd = [sys.executable, str(target), *sys.argv[1:], "--evaluate"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

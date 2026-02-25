import subprocess
import sys
from pathlib import Path


def main() -> int:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("MMYOLO shared prepare wrapper")
        print("Usage: python mmyolo/scripts/prepare_shared_dataset.py")
        print("This command runs shared/scripts/prepare_jet_dataset.py")
        return 0

    project_root = Path(__file__).resolve().parents[2]
    target = project_root / "shared" / "scripts" / "prepare_jet_dataset.py"
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

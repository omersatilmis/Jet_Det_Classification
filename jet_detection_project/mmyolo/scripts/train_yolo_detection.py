import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Train YOLO model with project defaults (MMYOLO)."
    )
    parser.add_argument(
        "--config",
        default=str(project_root / "mmyolo" / "configs" / "jet" / "yolov8_s_jet.py"),
        help="Path to MMYOLO config file.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(project_root / "work_dirs" / "yolov8_s_jet"),
        help="Training work directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in work directory.",
    )
    return parser.parse_known_args()


def main() -> int:
    args, extra_args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    target = project_root / "mmyolo" / "tools" / "train.py"

    cmd = [
        sys.executable,
        str(target),
        args.config,
        "--work-dir",
        args.work_dir,
    ]
    if args.resume:
        cmd.append("--resume")
    cmd.extend(extra_args)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

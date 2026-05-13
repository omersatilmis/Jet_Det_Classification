"""
Run MMDet and MMYOLO evaluators, then update AntJetDetUI eval_metrics.json.

Usage:
  python shared/scripts/run_eval_and_update.py --split val
  python shared/scripts/run_eval_and_update.py --split test --no-mmyolo
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "mmdetection").exists() and (p / "mmyolo").exists() and (p / "coco_annotations").exists():
            return p
    raise RuntimeError("Project root not found")


def run_cmd(cmd, cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run evals and update eval_metrics.json")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--no-mmdet", action="store_true")
    p.add_argument("--no-mmyolo", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--score-thr", type=float, default=None)
    p.add_argument("--iou-thr", type=float, default=None)

    p.add_argument("--mmdet-config", type=str, default=None)
    p.add_argument("--mmdet-checkpoint", type=str, default=None)
    p.add_argument("--mmdet-work-dir", type=str, default=None)
    p.add_argument("--mmdet-id", type=str, default="cascade-rcnn-r50-tiny")

    p.add_argument("--mmyolo-config", type=str, default=None)
    p.add_argument("--mmyolo-checkpoint", type=str, default=None)
    p.add_argument("--mmyolo-work-dir", type=str, default=None)
    p.add_argument("--mmyolo-id", type=str, default="yolov8-s-jet")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    python_exe = sys.executable

    outputs_dir = project_root / "outputs"
    mmdet_out = outputs_dir / f"eval_mmdet_{args.split}" / "coco_metrics.json"
    mmyolo_out = outputs_dir / f"eval_mmyolo_{args.split}" / "coco_metrics.json"

    if not args.no_mmdet:
        cmd = [
            python_exe,
            "mmdetection/evaluation/evaluate_mmdet.py",
            "--split",
            args.split,
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.score_thr is not None:
            cmd += ["--score-thr", str(args.score_thr)]
        if args.iou_thr is not None:
            cmd += ["--iou-thr", str(args.iou_thr)]
        if args.mmdet_config:
            cmd += ["--config", args.mmdet_config]
        if args.mmdet_checkpoint:
            cmd += ["--checkpoint", args.mmdet_checkpoint]
        if args.mmdet_work_dir:
            cmd += ["--work-dir", args.mmdet_work_dir]
        run_cmd(cmd, project_root)

    if not args.no_mmyolo:
        cmd = [
            python_exe,
            "mmyolo/evaluation/evaluate_mmyolo.py",
            "--split",
            args.split,
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.score_thr is not None:
            cmd += ["--score-thr", str(args.score_thr)]
        if args.iou_thr is not None:
            cmd += ["--iou-thr", str(args.iou_thr)]
        if args.mmyolo_config:
            cmd += ["--config", args.mmyolo_config]
        if args.mmyolo_checkpoint:
            cmd += ["--checkpoint", args.mmyolo_checkpoint]
        if args.mmyolo_work_dir:
            cmd += ["--work-dir", args.mmyolo_work_dir]
        run_cmd(cmd, project_root)

    update_cmd = [
        python_exe,
        "shared/scripts/update_eval_metrics.py",
    ]
    if not args.no_mmdet and mmdet_out.exists():
        update_cmd += ["--mmdet", str(mmdet_out), "--id-mmdet", args.mmdet_id]
    if not args.no_mmyolo and mmyolo_out.exists():
        update_cmd += ["--mmyolo", str(mmyolo_out), "--id-mmyolo", args.mmyolo_id]

    run_cmd(update_cmd, project_root)
    print("[OK] eval_metrics.json updated")


if __name__ == "__main__":
    main()

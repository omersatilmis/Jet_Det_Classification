import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [3407, 42, 1337]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 3-seed MMDetection training and report best seed by validation mAP"
    )
    parser.add_argument("--config", type=str, default="mmdetection/configs/cascade_rcnn_convnext_tiny.py")
    parser.add_argument("--work-dir", type=str, default="work_dirs/cascade_rcnn_convnext_tiny")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds, e.g. 3407,42,1337",
    )
    parser.add_argument("--prepare-coco", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _parse_seeds(seed_str: str) -> list[int]:
    values = []
    for item in seed_str.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("No valid seed values provided.")
    return values


def _find_latest_scalars(work_dir: Path) -> Path | None:
    candidates = sorted(work_dir.glob("**/vis_data/scalars.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _extract_best_map(scalars_path: Path) -> float | None:
    best = None
    try:
        with scalars_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                value = payload.get("coco/bbox_mAP")
                if value is None:
                    continue
                value = float(value)
                if best is None or value > best:
                    best = value
    except Exception:
        return None
    return best


def main() -> int:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)

    project_root = Path(__file__).resolve().parents[2]
    train_script = project_root / "mmdetection" / "scripts" / "train_jet_detection.py"

    summary = []

    for idx, seed in enumerate(seeds):
        run_work_dir = f"{args.work_dir}_seed{seed}"
        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            args.config,
            "--work-dir",
            run_work_dir,
            "--seed",
            str(seed),
        ]

        if args.deterministic:
            cmd.append("--deterministic")
        if args.resume:
            cmd.append("--resume")
        if args.fresh:
            cmd.append("--fresh")
        if args.prepare_coco and idx == 0:
            cmd.append("--prepare-coco")

        print("=" * 88)
        print(f"[INFO] Running seed={seed} | work_dir={run_work_dir}")
        print("[CMD]", " ".join(cmd))
        code = subprocess.call(cmd)

        if code != 0:
            print(f"[ERROR] Seed {seed} failed with exit code {code}")
            summary.append((seed, None, False))
            continue

        abs_run_dir = project_root / run_work_dir
        scalars = _find_latest_scalars(abs_run_dir)
        best_map = _extract_best_map(scalars) if scalars else None
        summary.append((seed, best_map, True))

    print("\n" + "=" * 88)
    print("[SUMMARY] Multi-seed training report")
    for seed, best_map, ok in summary:
        status = "OK" if ok else "FAILED"
        map_str = "n/a" if best_map is None else f"{best_map:.4f}"
        print(f"  - seed={seed:<6} status={status:<7} best_bbox_mAP={map_str}")

    valid = [(seed, score) for seed, score, ok in summary if ok and score is not None]
    if valid:
        best_seed, best_score = max(valid, key=lambda x: x[1])
        print(f"[BEST] seed={best_seed} with best_bbox_mAP={best_score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

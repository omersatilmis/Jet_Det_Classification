"""
Update AntJetDetUI eval_metrics.json from COCO evaluation outputs.

Usage:
  python shared/scripts/update_eval_metrics.py \
    --mmdet outputs/eval_mmdet_val/coco_metrics.json \
    --mmyolo outputs/eval_mmyolo_val/coco_metrics.json \
    --id-mmdet cascade-rcnn-convnext-tiny \
    --id-mmyolo yolov8-s-jet
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update eval_metrics.json from coco_metrics.json")
    p.add_argument("--mmdet", type=str, default=None, help="Path to MMDet coco_metrics.json")
    p.add_argument("--mmyolo", type=str, default=None, help="Path to MMYOLO coco_metrics.json")
    p.add_argument("--id-mmdet", type=str, default="cascade-rcnn-convnext-tiny")
    p.add_argument("--id-mmyolo", type=str, default="yolov8-s-jet")
    p.add_argument("--out", type=str, default=None, help="Output eval_metrics.json path")
    return p.parse_args()


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "AntJetDetUI").exists() and (p / "coco_annotations").exists():
            return p
    raise RuntimeError("Project root not found")


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _find_metric(metrics: Dict[str, Any], keys, fallback_contains) -> Optional[float]:
    for k in keys:
        if k in metrics:
            return _to_float(metrics[k])
    for k, v in metrics.items():
        if any(s in k for s in fallback_contains):
            return _to_float(v)
    return None


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_avg_iou(metrics_path: Path) -> Optional[float]:
    avg_iou_path = metrics_path.parent / "avg_iou.json"
    if not avg_iou_path.exists():
        return None
    try:
        with open(avg_iou_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _to_float(data.get("avg_iou"))
    except Exception:
        return None


def _load_pr_curve(metrics_path: Path) -> Optional[list]:
    pr_path = metrics_path.parent / "pr_curve.json"
    if not pr_path.exists():
        return None
    try:
        with open(pr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def update_entry(store: Dict[str, Any], model_id: str, metrics: Dict[str, Any], metrics_path: Path) -> None:
    # COCO mAP (AP50:95)
    map_value = _find_metric(
        metrics,
        keys=["coco/bbox_mAP", "bbox_mAP", "mAP", "coco/bbox_mAP_50_95"],
        fallback_contains=["bbox_mAP", "mAP_50_95", "mAP"],
    )

    # Use AP50 as IoU proxy
    iou_value = _find_metric(
        metrics,
        keys=["coco/bbox_mAP_50", "bbox_mAP_50", "AP50", "ap50"],
        fallback_contains=["mAP_50", "AP50"],
    )

    mean_iou = _load_avg_iou(metrics_path)
    pr_curve = _load_pr_curve(metrics_path)
    store[model_id] = {
        "map": map_value,
        "iou": iou_value,
        "mean_iou": mean_iou,
        "pr_curve": pr_curve,
    }


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    out_path = Path(args.out) if args.out else project_root / "AntJetDetUI" / "backend" / "eval_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_store: Dict[str, Any] = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                metrics_store = json.load(f)
            except Exception:
                metrics_store = {}

    if args.mmdet:
        mmdet_path = Path(args.mmdet)
        mmdet_metrics = load_metrics(mmdet_path)
        update_entry(metrics_store, args.id_mmdet, mmdet_metrics, mmdet_path)

    if args.mmyolo:
        mmyolo_path = Path(args.mmyolo)
        mmyolo_metrics = load_metrics(mmyolo_path)
        update_entry(metrics_store, args.id_mmyolo, mmyolo_metrics, mmyolo_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_store, f, indent=2, ensure_ascii=False)

    print(f"[OK] Updated: {out_path}")


if __name__ == "__main__":
    main()

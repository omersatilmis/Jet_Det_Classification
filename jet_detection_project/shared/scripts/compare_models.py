"""
Jet Detection – Model Karşılaştırma Aracı
==========================================
MMDetection ve MMYOLO modellerinin değerlendirme sonuçlarını yan yana
karşılaştırır ve bir özet rapor oluşturur.

Önce her iki modeli de ayrı ayrı değerlendirin:
    python mmdetection/evaluation/evaluate_mmdet.py --split val
    python mmyolo/evaluation/evaluate_mmyolo.py --split val

Ardından karşılaştırma yapın:
    python shared/scripts/compare_models.py
    python shared/scripts/compare_models.py --split test
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Model Karşılaştırma Aracı")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--mmdet-dir", type=str, default=None,
                   help="MMDet eval çıktı dizini")
    p.add_argument("--mmyolo-dir", type=str, default=None,
                   help="MMYOLO eval çıktı dizini")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "mmdetection").exists() and (p / "mmyolo").exists():
            return p
    raise RuntimeError("Proje kökü bulunamadı")


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_comparison_bar(mmdet_metrics, yolo_metrics, metric_keys, labels,
                        title, out_path):
    """İki modeli yan yana bar chart ile karşılaştırır."""
    x = np.arange(len(metric_keys))
    width = 0.35

    mmdet_vals = [mmdet_metrics.get(k, 0) for k in metric_keys]
    yolo_vals = [yolo_metrics.get(k, 0) for k in metric_keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, mmdet_vals, width, label="Cascade R-CNN (MMDet)",
                   color="#3B82F6", edgecolor="white")
    bars2 = ax.bar(x + width/2, yolo_vals, width, label="YOLOv8-S (MMYOLO)",
                   color="#F97316", edgecolor="white")

    ax.set_ylabel("Değer")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Bar üstü değer
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_per_class_comparison(mmdet_pc, yolo_pc, out_path):
    """Per-class F1 karşılaştırma bar chart."""
    classes = [r["Class"] for r in mmdet_pc]
    mmdet_f1 = [r["F1"] for r in mmdet_pc]
    yolo_f1 = [r["F1"] for r in yolo_pc] if yolo_pc else [0] * len(classes)

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, mmdet_f1, width, label="Cascade R-CNN", color="#3B82F6")
    ax.bar(x + width/2, yolo_f1, width, label="YOLOv8-S", color="#F97316")
    ax.set_ylabel("F1 Score")
    ax.set_title("Sınıf Bazlı F1 Karşılaştırması")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.0)

    for i, (v1, v2) in enumerate(zip(mmdet_f1, yolo_f1)):
        ax.text(i - width/2, v1 + 0.02, f"{v1:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, v2 + 0.02, f"{v2:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    project_root = find_project_root()

    mmdet_dir = Path(args.mmdet_dir) if args.mmdet_dir else \
        project_root / "outputs" / f"eval_mmdet_{args.split}"
    yolo_dir = Path(args.mmyolo_dir) if args.mmyolo_dir else \
        project_root / "outputs" / f"eval_mmyolo_{args.split}"
    out_dir = Path(args.out_dir) if args.out_dir else \
        project_root / "outputs" / f"comparison_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] MMDet eval  : {mmdet_dir}")
    print(f"[INFO] MMYOLO eval : {yolo_dir}")
    print(f"[INFO] Output      : {out_dir}")

    # Veri yükle
    mmdet_coco = load_json(mmdet_dir / "coco_metrics.json") or {}
    yolo_coco = load_json(yolo_dir / "coco_metrics.json") or {}
    mmdet_pc = load_json(mmdet_dir / "per_class_metrics.json") or []
    yolo_pc = load_json(yolo_dir / "per_class_metrics.json") or []

    has_mmdet = bool(mmdet_coco)
    has_yolo = bool(yolo_coco)

    if not has_mmdet and not has_yolo:
        print("[ERROR] Her iki modelin de eval çıktısı bulunamadı!")
        print(f"  Beklenen: {mmdet_dir / 'coco_metrics.json'}")
        print(f"  Beklenen: {yolo_dir / 'coco_metrics.json'}")
        sys.exit(1)

    # ========== COCO Metrikleri Karşılaştırması ==========
    print("\n" + "=" * 70)
    print("MODEL KARŞILAŞTIRMASI")
    print("=" * 70)

    coco_keys = ["coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
                 "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l"]
    coco_labels = ["mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"]

    print(f"\n  {'Metrik':<12} {'Cascade R-CNN':>14} {'YOLOv8-S':>14} {'Fark':>10}")
    print("  " + "-" * 52)

    comparison_report = {"split": args.split, "coco_comparison": [], "per_class_comparison": []}

    for key, label in zip(coco_keys, coco_labels):
        v1 = mmdet_coco.get(key, 0.0) if has_mmdet else "-"
        v2 = yolo_coco.get(key, 0.0) if has_yolo else "-"
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v2 - v1
            diff_str = f"{diff:>+.4f}"
        else:
            diff_str = "N/A"

        v1_str = f"{v1:.4f}" if isinstance(v1, (int, float)) else v1
        v2_str = f"{v2:.4f}" if isinstance(v2, (int, float)) else v2
        print(f"  {label:<12} {v1_str:>14} {v2_str:>14} {diff_str:>10}")

        comparison_report["coco_comparison"].append(
            dict(metric=label, cascade_rcnn=v1, yolov8=v2)
        )

    # ========== Per-Class Karşılaştırma ==========
    if mmdet_pc and yolo_pc:
        print(f"\n  {'Sınıf':<8} {'CascadeRCNN F1':>16} {'YOLOv8 F1':>12} {'Fark':>10}")
        print("  " + "-" * 48)

        for m, y in zip(mmdet_pc, yolo_pc):
            diff = y["F1"] - m["F1"]
            print(f"  {m['Class']:<8} {m['F1']:>16.4f} {y['F1']:>12.4f} {diff:>+10.4f}")
            comparison_report["per_class_comparison"].append(
                dict(cls=m["Class"], cascade_f1=m["F1"], yolo_f1=y["F1"])
            )

    # ========== Grafikler ==========
    if has_mmdet and has_yolo:
        plot_comparison_bar(mmdet_coco, yolo_coco, coco_keys, coco_labels,
                            f"COCO Metrikleri Karşılaştırması ({args.split})",
                            out_dir / "coco_comparison.png")
        print(f"\n  → COCO karşılaştırma grafiği: {out_dir / 'coco_comparison.png'}")

    if mmdet_pc and yolo_pc:
        plot_per_class_comparison(mmdet_pc, yolo_pc,
                                  out_dir / "per_class_f1_comparison.png")
        print(f"  → Per-class F1 grafiği: {out_dir / 'per_class_f1_comparison.png'}")

    # Rapor kaydet
    report_path = out_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  → Karşılaştırma raporu: {report_path}")

    # ========== Sonuç ==========
    if has_mmdet and has_yolo:
        mmdet_map = mmdet_coco.get("coco/bbox_mAP", 0)
        yolo_map = yolo_coco.get("coco/bbox_mAP", 0)
        winner = "Cascade R-CNN (MMDet)" if mmdet_map >= yolo_map else "YOLOv8-S (MMYOLO)"
        print(f"\n  🏆 En iyi model: {winner} (mAP: {max(mmdet_map, yolo_map):.4f})")

    print("\n" + "=" * 70)
    print("KARŞILAŞTIRMA TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    main()

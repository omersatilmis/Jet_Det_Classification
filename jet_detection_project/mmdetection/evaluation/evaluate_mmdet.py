"""
Jet Detection – MMDetection Model Evaluator
============================================
Cascade R-CNN + ConvNeXt-Tiny modelini val/test seti üzerinde değerlendirir.

Çıktılar:
  - COCO metrikleri (mAP, AP50, AP75, AR, per-class AP)
  - Confusion matrix (görsel + CSV)
  - Sınıf bazlı precision/recall tablosu
  - Hatalı tespit örnekleri (FP/FN görselleştirme)

Kullanım:
    python mmdetection/evaluation/evaluate_mmdet.py
    python mmdetection/evaluation/evaluate_mmdet.py --split test
    python mmdetection/evaluation/evaluate_mmdet.py --tta --score-thr 0.4
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="MMDetection Cascade R-CNN Evaluator")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--config", type=str, default=None,
                   help="Config dosyası (varsayılan: mmdetection/configs/cascade_rcnn_convnext_tiny.py)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint dosyası (varsayılan: work_dir altında best_*.pth aranır)")
    p.add_argument("--work-dir", type=str, default=None,
                   help="Checkpoint arama dizini")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--score-thr", type=float, default=0.3,
                   help="Confusion matrix ve görselleştirme için skor eşiği")
    p.add_argument("--iou-thr", type=float, default=0.5,
                   help="TP/FP belirleme IoU eşiği")
    p.add_argument("--tta", action="store_true",
                   help="Test-Time Augmentation uygula")
    p.add_argument("--max-vis", type=int, default=20,
                   help="Hatalı tespit görselleştirme sayısı (0=kapalı)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Çıktı dizini (varsayılan: outputs/eval_mmdet_<split>)")
    return p.parse_args()


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "mmdetection").exists() and (p / "coco_annotations").exists():
            return p
    raise RuntimeError("Proje kökü bulunamadı")


def find_best_checkpoint(work_dir: Path) -> Path | None:
    """best_*.pth > latest.pth > epoch_*.pth (en büyük epoch)"""
    import re

    candidates = list(work_dir.glob("best_*.pth"))
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    latest = work_dir / "latest.pth"
    if latest.exists():
        return latest

    epoch_ckpts = list(work_dir.glob("epoch_*.pth"))
    if epoch_ckpts:
        def epoch_num(p):
            m = re.search(r"epoch_(\d+)\.pth$", p.name)
            return int(m.group(1)) if m else -1
        return max(epoch_ckpts, key=epoch_num)

    return None


def compute_iou(box_a, box_b):
    """box format: [x1, y1, x2, y2]"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_confusion_data(model, dataset, class_names, score_thr, iou_thr, device):
    """Her görsel için TP/FP/FN hesaplar, confusion matrix verisini döndürür."""
    from mmdet.apis import inference_detector

    n_cls = len(class_names)
    # confusion[pred_cls][gt_cls] — son satır/sütun: background
    confusion = np.zeros((n_cls + 1, n_cls + 1), dtype=int)

    fp_examples = []
    fn_examples = []

    for idx in range(len(dataset)):
        data_info = dataset.get_data_info(idx)
        img_path = data_info.get("img_path", "")

        # GT
        gt_instances = data_info.get("instances", [])
        gt_boxes = [inst["bbox"] for inst in gt_instances]  # x1,y1,w,h → x1,y1,x2,y2
        gt_boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes]
        gt_labels = [inst["bbox_label"] for inst in gt_instances]
        gt_matched = [False] * len(gt_boxes)

        # Pred
        result = inference_detector(model, img_path)
        pred_instances = result.pred_instances
        keep = pred_instances.scores >= score_thr
        pred_boxes = pred_instances.bboxes[keep].cpu().numpy()
        pred_labels = pred_instances.labels[keep].cpu().numpy()
        pred_scores = pred_instances.scores[keep].cpu().numpy()

        # Match
        for pi, (pb, pl, ps) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            best_iou = 0
            best_gi = -1
            for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_thr and best_gi >= 0:
                gt_matched[best_gi] = True
                confusion[int(pl)][gt_labels[best_gi]] += 1
                if int(pl) != gt_labels[best_gi] and len(fp_examples) < 50:
                    fp_examples.append(dict(img=img_path, pred=int(pl),
                                            gt=gt_labels[best_gi], score=float(ps)))
            else:
                # FP → background column
                confusion[int(pl)][n_cls] += 1
                if len(fp_examples) < 50:
                    fp_examples.append(dict(img=img_path, pred=int(pl),
                                            gt=-1, score=float(ps)))

        for gi, matched in enumerate(gt_matched):
            if not matched:
                confusion[n_cls][gt_labels[gi]] += 1
                if len(fn_examples) < 50:
                    fn_examples.append(dict(img=img_path, gt=gt_labels[gi]))

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(dataset)}] confusion matrix hesaplanıyor...")

    return confusion, fp_examples, fn_examples


def plot_confusion_matrix(confusion, class_names, out_path):
    import matplotlib.pyplot as plt

    labels = list(class_names) + ["BG"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel="Predicted", xlabel="Ground Truth",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = confusion.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(confusion[i, j]),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black",
                    fontsize=10)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  → Confusion matrix kaydedildi: {out_path}")


def compute_per_class_metrics(confusion, class_names):
    """Confusion matrix'ten per-class precision/recall hesapla."""
    rows = []
    n_cls = len(class_names)
    for i in range(n_cls):
        tp = confusion[i][i]
        fp = sum(confusion[i]) - tp
        fn = sum(confusion[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(dict(Class=class_names[i], TP=tp, FP=fp, FN=fn,
                         Precision=round(precision, 4), Recall=round(recall, 4),
                         F1=round(f1, 4)))
    return rows


def main():
    args = parse_args()
    project_root = find_project_root()
    print(f"[INFO] Project root: {project_root}")

    # --- Yolları belirle ---
    config_path = Path(args.config) if args.config else \
        project_root / "mmdetection" / "configs" / "cascade_rcnn_convnext_tiny.py"

    if args.work_dir:
        work_dir = Path(args.work_dir) if Path(args.work_dir).is_absolute() \
            else project_root / args.work_dir
    else:
        work_dir = project_root / "work_dirs" / "mmdetection" / config_path.stem

        # Fallback: eski dizin yapısı
        if not work_dir.exists():
            work_dir = project_root / "work_dirs" / config_path.stem
        if not work_dir.exists():
            work_dir = project_root / "work_dirs" / "cascade_rcnn_r50_tiny"

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() \
            else project_root / args.checkpoint
    else:
        ckpt_path = find_best_checkpoint(work_dir)

    if ckpt_path is None or not ckpt_path.exists():
        print(f"[ERROR] Checkpoint bulunamadı! work_dir: {work_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else \
        project_root / "outputs" / f"eval_mmdet_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Config   : {config_path}")
    print(f"[INFO] Checkpoint: {ckpt_path}")
    print(f"[INFO] Split     : {args.split}")
    print(f"[INFO] Output    : {out_dir}")

    # --- MMDet yükle ---
    from mmengine.config import Config, ConfigDict
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules
    register_all_modules()

    cfg = Config.fromfile(str(config_path))

    # Path yamalama
    archive_dir = project_root.parent / "archive"
    images_dir = archive_dir / "dataset"
    ann_dir = project_root / "coco_annotations"

    if not images_dir.exists():
        images_dir = project_root / "archive" / "dataset"

    ann_map = {"val": "instances_validation.json", "test": "instances_test.json"}

    for mode in ["val", "test"]:
        dl_key = f"{mode}_dataloader"
        dl = getattr(cfg, dl_key, None)
        if dl is None:
            continue

        # ClassBalancedDataset vs düz dataset
        ds = dl.dataset
        while hasattr(ds, "dataset"):
            ds = ds.dataset

        ds.data_root = ""
        ds.data_prefix = dict(img=str(images_dir) + os.sep)
        ds.ann_file = str(ann_dir / ann_map[mode])

    if hasattr(cfg, "val_evaluator"):
        cfg.val_evaluator.ann_file = str(ann_dir / ann_map.get("val", "instances_validation.json"))
    if hasattr(cfg, "test_evaluator"):
        cfg.test_evaluator.ann_file = str(ann_dir / ann_map.get("test", "instances_test.json"))

    # Windows fix
    for mode in ["train", "val", "test"]:
        dl = getattr(cfg, f"{mode}_dataloader", None)
        if dl:
            dl.num_workers = 0
            dl.persistent_workers = False

    cfg.load_from = str(ckpt_path)
    cfg.work_dir = str(out_dir)

    # TTA
    if args.tta and hasattr(cfg, "tta_model") and hasattr(cfg, "tta_pipeline"):
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        dl_key = f"{args.split}_dataloader" if args.split == "test" else "val_dataloader"
        dl = getattr(cfg, dl_key, None)
        if dl:
            ds = dl.dataset
            while hasattr(ds, "dataset"):
                ds = ds.dataset
            ds.pipeline = cfg.tta_pipeline
        print("[INFO] TTA aktif")

    # ========== BÖLÜM 1: COCO Metrics (Runner ile) ==========
    print("\n" + "=" * 70)
    print("BÖLÜM 1: COCO METRİKLERİ")
    print("=" * 70)

    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)

    if args.split == "test":
        metrics = runner.test()
    else:
        metrics = runner.val()

    # Metrikleri kaydet
    metrics_file = out_dir / "coco_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"  → COCO metrikleri: {metrics_file}")

    # ========== BÖLÜM 2: Confusion Matrix ==========
    print("\n" + "=" * 70)
    print("BÖLÜM 2: CONFUSION MATRIX & PER-CLASS METRİKLER")
    print("=" * 70)

    model = init_detector(str(config_path), str(ckpt_path), device=args.device)

    # Dataset oluştur
    from mmdet.datasets import CocoDataset
    class_names = ("F16", "F18", "F22", "F35")
    ann_file = str(ann_dir / ann_map[args.split])

    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(1600, 1000), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs'),
    ]

    dataset = CocoDataset(
        ann_file=ann_file,
        data_root="",
        data_prefix=dict(img=str(images_dir) + os.sep),
        metainfo=dict(classes=class_names),
        test_mode=True,
        pipeline=test_pipeline,
    )

    print(f"  Dataset yüklendi: {len(dataset)} görsel ({args.split})")

    confusion, fp_examples, fn_examples = build_confusion_data(
        model, dataset, class_names, args.score_thr, args.iou_thr, args.device
    )

    # Confusion Matrix — Görsel
    plot_confusion_matrix(confusion, class_names, out_dir / "confusion_matrix.png")

    # Confusion Matrix — CSV
    import csv
    labels = list(class_names) + ["BG"]
    csv_path = out_dir / "confusion_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pred \\ GT"] + labels)
        for i, row in enumerate(confusion):
            writer.writerow([labels[i]] + list(row))
    print(f"  → Confusion matrix CSV: {csv_path}")

    # Per-class metrics
    per_class = compute_per_class_metrics(confusion, class_names)
    print("\n  Per-Class Metrics (score_thr={}, iou_thr={}):".format(args.score_thr, args.iou_thr))
    print(f"  {'Class':<8} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 50)
    for r in per_class:
        print(f"  {r['Class']:<8} {r['TP']:>5} {r['FP']:>5} {r['FN']:>5} "
              f"{r['Precision']:>8.4f} {r['Recall']:>8.4f} {r['F1']:>8.4f}")

    pc_path = out_dir / "per_class_metrics.json"
    with open(pc_path, "w", encoding="utf-8") as f:
        json.dump(per_class, f, indent=2, ensure_ascii=False)
    print(f"\n  → Per-class metrikleri: {pc_path}")

    # Hatalı örnekler JSON
    errors_path = out_dir / "error_examples.json"
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(dict(false_positives=fp_examples[:args.max_vis],
                       false_negatives=fn_examples[:args.max_vis]),
                  f, indent=2, ensure_ascii=False, default=str)
    print(f"  → Hatalı örnek listesi: {errors_path}")

    # ========== BÖLÜM 3: Özet ==========
    print("\n" + "=" * 70)
    print("DEĞERLENDİRME TAMAMLANDI")
    print("=" * 70)
    print(f"  Model    : Cascade R-CNN + ConvNeXt-Tiny")
    print(f"  Split    : {args.split}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Çıktılar : {out_dir}")
    print(f"  TTA      : {'Evet' if args.tta else 'Hayır'}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import random
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

try:
    import cv2
except ImportError:
    cv2 = None


def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_coco_size_category(area: float) -> str:
    """COCO size categories: Small (< 32^2), Medium, Large (> 96^2)"""
    if area < 32 * 32:
        return "Small"
    elif area < 96 * 96:
        return "Medium"
    else:
        return "Large"


def coco_stats(coco: dict) -> dict:
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}
    img_id_to_size = {im["id"]: (im.get("width", 0), im.get("height", 0)) for im in images}

    img_count = len(images)
    ann_count = len(anns)

    # Basic counters
    ann_per_img = Counter([a["image_id"] for a in anns])
    class_counts = Counter()
    class_images = defaultdict(set)
    
    # Size analysis
    size_categories = Counter()
    class_size_dist = defaultdict(Counter)
    
    # Geometry analysis
    bbox_wh = []
    bbox_areas = []
    aspect_ratios = []
    rel_areas = []
    
    # Spatial analysis
    spatial_coords = defaultdict(list)  # class -> list of normalized centers (cx, cy)

    for a in anns:
        cid = a["category_id"]
        cname = cat_id_to_name.get(cid, str(cid))
        class_counts[cname] += 1
        class_images[cname].add(a["image_id"])

        x, y, w, h = a["bbox"]
        area = w * h
        bbox_areas.append(area)
        bbox_wh.append((w, h))
        
        # Aspect Ratio
        if h > 0:
            aspect_ratios.append(w / h)
            
        # Size category
        sz_cat = get_coco_size_category(area)
        size_categories[sz_cat] += 1
        class_size_dist[cname][sz_cat] += 1
        
        # Relative area
        w_img, h_img = img_id_to_size.get(a["image_id"], (0, 0))
        if w_img > 0 and h_img > 0:
            rel_areas.append(area / (w_img * h_img))
            # Spatial center
            cx = (x + w/2) / w_img
            cy = (y + h/2) / h_img
            spatial_coords[cname].append((cx, cy))

    # Image sizes
    widths = np.array([im.get("width", 0) for im in images])
    heights = np.array([im.get("height", 0) for im in images])

    return {
        "img_count": img_count,
        "ann_count": ann_count,
        "class_counts": dict(class_counts),
        "images_per_class": {k: len(v) for k, v in class_images.items()},
        "ann_per_img": dict(ann_per_img),
        "img_widths": widths.tolist(),
        "img_heights": heights.tolist(),
        "bbox_areas": bbox_areas,
        "bbox_wh": bbox_wh,
        "aspect_ratios": aspect_ratios,
        "rel_areas": rel_areas,
        "size_categories": dict(size_categories),
        "class_size_dist": {k: dict(v) for k, v in class_size_dist.items()},
        "spatial_coords": {k: v for k, v in spatial_coords.items()},
        "categories": [cat_id_to_name[c["id"]] for c in cats],
    }


def plot_spatial_heatmap(stats: dict, split: str, out_dir: Path):
    """Generates a spatial heatmap for the entire split and per-class."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # All classes combined
    all_coords = []
    for cls_coords in stats["spatial_coords"].values():
        all_coords.extend(cls_coords)
    
    if not all_coords:
        return

    def create_heatmap(coords, title, filename):
        coords = np.array(coords)
        plt.figure(figsize=(8, 7))
        h, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=20, range=[[0, 1], [0, 1]])
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]  # Flip Y for image coordinates
        
        plt.imshow(h.T, extent=extent, interpolation='gaussian', cmap='jet')
        plt.colorbar(label='Density')
        plt.title(f"{title} ({split})")
        plt.xlabel("X (Normalized)")
        plt.ylabel("Y (Normalized)")
        plt.grid(alpha=0.3)
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()

    create_heatmap(all_coords, "Spatial Distribution - All Classes", f"spatial_heatmap_{split}_all.png")
    
    # Per class heatmap (top 4 if many)
    for cls, coords in stats["spatial_coords"].items():
        if len(coords) > 5:
            create_heatmap(coords, f"Heatmap - {cls}", f"spatial_heatmap_{split}_{cls}.png")


def plot_split_comparison(all_reports: dict, out_dir: Path):
    """Compares class distribution across splits."""
    splits = list(all_reports.keys())
    if not splits: return
    
    classes = all_reports[splits[0]]["categories"]
    n_classes = len(classes)
    n_splits = len(splits)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_classes)
    width = 0.8 / n_splits
    
    for i, split in enumerate(splits):
        counts = [all_reports[split]["class_counts"].get(cls, 0) for cls in classes]
        ax.bar(x + i*width, counts, width, label=split)
        
    ax.set_ylabel('Annotations')
    ax.set_title('Class Distribution Across Splits')
    ax.set_xticks(x + width * (n_splits-1) / 2)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "split_comparison.png", dpi=200)
    plt.close()


def generate_diagnostics(all_reports: dict) -> list:
    """Analyzes the stats and provides 'Reis' advice."""
    advice = []
    
    for split, stats in all_reports.items():
        advice.append(f"--- {split.upper()} DIAGNOSTICS ---")
        
        # 1. Imbalance Check
        counts = stats["class_counts"]
        if counts:
            max_c = max(counts.values())
            min_c = min(counts.values())
            if max_c > min_c * 5:
                minority = [k for k, v in counts.items() if v == min_c][0]
                advice.append(f"⚠️ Dengesizlik: {minority} sınıfı çok az ({min_c} adet). Model bu sınıfı öğrenmekte zorlanabilir.")
        
        # 2. Small target check
        small_count = stats["size_categories"].get("Small", 0)
        total_ann = stats["ann_count"]
        if total_ann > 0:
            small_ratio = small_count / total_ann
            if small_ratio > 0.5:
                advice.append(f"🔍 Küçük Hedef Yoğunluğu: Nesnelerin %{small_ratio*100:.1f}'i 'Small' kategorisinde. SAHI veya yüksek çözünürlüklü (800+) eğitim önerilir.")

        # 3. Aspect ratio check
        ars = np.array(stats["aspect_ratios"])
        if ars.size > 0:
            extreme = np.sum((ars > 3) | (ars < 0.33))
            if extreme > total_ann * 0.1:
                advice.append(f"📏 Ekstrem Oranlar: Nesnelerin %{extreme/total_ann*100:.1f}'i çok uzun veya çok geniş. 'YOLOv5RandomAffine' gibi augmentasyonlar kritik.")

    return advice


def main():
    ap = argparse.ArgumentParser(description="Advanced Dataset Analysis ('Reis' Upgrade)")
    ap.add_argument("--ann-dir", default="coco_annotations", help="COCO json klasörü")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Analiz edilecek splitler")
    ap.add_argument("--out-dir", default="dataset_analysis", help="Çıktı klasörü")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ann_dir = repo_root / args.ann_dir
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_to_file = {
        "train": "instances_train.json",
        "validation": "instances_validation.json",
        "val": "instances_validation.json",
        "test": "instances_test.json",
    }

    all_reports = {}

    for split in args.splits:
        if split not in split_to_file: continue
        
        json_path = ann_dir / split_to_file[split]
        if not json_path.exists():
            print(f"Skipping {split}, file not found: {json_path}")
            continue

        print(f"🚀 Analyzing {split}...")
        coco = load_coco(json_path)
        stats = coco_stats(coco)
        all_reports[split] = stats
        
        # Plots
        plot_spatial_heatmap(stats, split, out_dir / "heatmaps")
        
        # Size categories bar plot
        plt.figure(figsize=(8, 5))
        scat = stats["size_categories"]
        plt.bar(scat.keys(), scat.values(), color=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title(f"COCO Size Distribution - {split}")
        plt.savefig(out_dir / f"size_dist_{split}.png")
        plt.close()

    # Comparison and Global Analysis
    if all_reports:
        plot_split_comparison(all_reports, out_dir)
        
        # Diagnostics
        advice = generate_diagnostics(all_reports)
        print("\n" + "!"*40)
        print("REİS'İN TAVSİYELERİ (DIAGNOSTICS)")
        print("!"*40)
        for line in advice:
            print(line)
            
        with open(out_dir / "diagnostics.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(advice))

    # Save upgraded JSON (stripping huge lists for readability)
    final_json = {}
    for k, v in all_reports.items():
        report = v.copy()
        for huge_key in ["img_widths", "img_heights", "bbox_areas", "bbox_wh", "aspect_ratios", "rel_areas", "spatial_coords"]:
            del report[huge_key]
        final_json[k] = report

    with open(out_dir / "dataset_analysis_report_v2.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    print(f"\n✅ Analiz tamamlandı. Raporlar: {out_dir}")


if __name__ == "__main__":
    main()

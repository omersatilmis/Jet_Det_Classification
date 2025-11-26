import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    cv2 = None


def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coco_stats(coco: dict) -> dict:
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}

    img_count = len(images)
    ann_count = len(anns)

    # annotations per image
    ann_per_img = Counter([a["image_id"] for a in anns])

    # class counts (annotation count)
    class_counts = Counter()
    bbox_wh = []   # (w, h)
    bbox_area = []

    # images per class (unique image count per class)
    class_images = defaultdict(set)

    for a in anns:
        cid = a["category_id"]
        cname = cat_id_to_name.get(cid, str(cid))

        class_counts[cname] += 1

        # images per class
        class_images[cname].add(a["image_id"])

        x, y, w, h = a["bbox"]
        bbox_wh.append((w, h))
        bbox_area.append(w * h)

    images_per_class = {k: len(v) for k, v in class_images.items()}

    bbox_wh = np.array(bbox_wh, dtype=np.float32) if bbox_wh else np.zeros((0, 2), dtype=np.float32)
    bbox_area = np.array(bbox_area, dtype=np.float32) if bbox_area else np.zeros((0,), dtype=np.float32)

    # image sizes
    widths = np.array([im.get("width", 0) for im in images], dtype=np.int32) if images else np.zeros((0,), dtype=np.int32)
    heights = np.array([im.get("height", 0) for im in images], dtype=np.int32) if images else np.zeros((0,), dtype=np.int32)

    # bbox relative area
    img_id_to_size = {im["id"]: (im.get("width", 0), im.get("height", 0)) for im in images}
    rel_area = []
    for a in anns:
        w_img, h_img = img_id_to_size.get(a["image_id"], (0, 0))
        if w_img > 0 and h_img > 0:
            rel_area.append((a["bbox"][2] * a["bbox"][3]) / (w_img * h_img))
    rel_area = np.array(rel_area, dtype=np.float32) if rel_area else np.zeros((0,), dtype=np.float32)

    return {
        "img_count": img_count,
        "ann_count": ann_count,
        "class_counts": dict(class_counts),
        "images_per_class": dict(images_per_class),
        "ann_per_img": ann_per_img,
        "img_widths": widths,
        "img_heights": heights,
        "bbox_wh": bbox_wh,
        "bbox_area": bbox_area,
        "bbox_rel_area": rel_area,
        "categories": [cat_id_to_name[c["id"]] for c in cats],
    }


def describe_arr(name: str, arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"{name}: (boş)"
    return (
        f"{name}: n={arr.size}, "
        f"min={float(np.min(arr)):.4f}, "
        f"p25={float(np.percentile(arr, 25)):.4f}, "
        f"median={float(np.median(arr)):.4f}, "
        f"p75={float(np.percentile(arr, 75)):.4f}, "
        f"max={float(np.max(arr)):.4f}, "
        f"mean={float(np.mean(arr)):.4f}"
    )


def plot_class_counts(title: str, class_counts: dict, out_path: Path):
    if not class_counts:
        return

    names = list(class_counts.keys())
    vals = [class_counts[k] for k in names]

    plt.figure(figsize=(8, 4))
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist(title: str, arr: np.ndarray, out_path: Path, bins=50, xlabel="Value"):
    if arr.size == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def visualize_examples(coco: dict, img_dir: Path, out_dir: Path, max_images: int = 5, seed: int = 42):
    """
    Saves a few example images with GT bounding boxes drawn on top.
    Expects: img_dir / im["file_name"] to exist.
    """
    if cv2 is None:
        print("⚠️ OpenCV (cv2) bulunamadı. Örnek görseller çizilmeyecek. Kur: pip install opencv-python")
        return

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}

    if not images:
        return

    img_id_to_anns = defaultdict(list)
    for a in anns:
        img_id_to_anns[a["image_id"]].append(a)

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    samples = rng.sample(images, min(max_images, len(images)))

    saved = 0
    for im in samples:
        file_name = im.get("file_name")
        if not file_name:
            continue

        img_path = img_dir / file_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for a in img_id_to_anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            label = cats.get(a["category_id"], "unknown")

            # rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # label background and text
            text = str(label)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y0 = max(0, y - th - 8)
            cv2.rectangle(img, (x, y0), (x + tw + 6, y0 + th + 6), (0, 255, 0), -1)
            cv2.putText(img, text, (x + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_path = out_dir / f"example_{Path(file_name).stem}_id{im['id']}.jpg"
        cv2.imwrite(str(out_path), img)
        saved += 1

    print(f"🖼️ Örnek görseller kaydedildi: {out_dir} (saved={saved})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="coco_annotations", help="COCO json klasörü (projeye göre relatif)")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                    help="Analiz edilecek splitler: train validation test")
    ap.add_argument("--out-dir", default="dataset_analysis", help="Rapor+grafik çıktı klasörü")

    # NEW: image dir for example visualizations
    ap.add_argument("--img-dir", default=None,
                    help="Görsel kök klasörü. Eğer vermezsen örnek bbox çizimi atlanır. "
                         "Örn: images veya data/images. Split alt klasörü varsa otomatik dener.")
    ap.add_argument("--num-examples", type=int, default=5, help="Her split için çizilecek örnek görsel sayısı")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (örnek görsel seçimi için)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    ann_dir = Path(args.ann_dir)
    if not ann_dir.is_absolute():
        ann_dir = repo_root / ann_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir_root = None
    if args.img_dir is not None:
        img_dir_root = Path(args.img_dir)
        if not img_dir_root.is_absolute():
            img_dir_root = repo_root / img_dir_root

    split_to_file = {
        "train": "instances_train.json",
        "validation": "instances_validation.json",
        "val": "instances_validation.json",
        "test": "instances_test.json",
    }

    all_reports = {}

    for split in args.splits:
        if split not in split_to_file:
            raise ValueError(f"Split tanınmadı: {split}. Kullan: train/validation/test")

        json_path = ann_dir / split_to_file[split]
        if not json_path.exists():
            raise FileNotFoundError(f"COCO json bulunamadı: {json_path}")

        coco = load_coco(json_path)
        stats = coco_stats(coco)
        all_reports[split] = stats

        print("\n" + "=" * 80)
        print(f"SPLIT: {split}  |  FILE: {json_path.name}")
        print("=" * 80)
        print(f"Images: {stats['img_count']}")
        print(f"Annotations (bboxes): {stats['ann_count']}")

        # class distribution (annotation count)
        print("\nClass counts (annotations):")
        for k, v in sorted(stats["class_counts"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {k}: {v}")

        # images per class
        print("\nImages per class (unique images containing that class):")
        for k, v in sorted(stats["images_per_class"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {k}: {v}")

        # annotations per image
        api = np.array(list(stats["ann_per_img"].values()), dtype=np.int32)
        print("\nAnnotations per image:")
        print(describe_arr("ann/img", api.astype(np.float32)))

        # image size
        if stats["img_widths"].size:
            print("\nImage sizes:")
            print(describe_arr("width", stats["img_widths"].astype(np.float32)))
            print(describe_arr("height", stats["img_heights"].astype(np.float32)))

        # bbox stats
        if stats["bbox_wh"].shape[0]:
            w = stats["bbox_wh"][:, 0]
            h = stats["bbox_wh"][:, 1]
            print("\nBBox sizes (pixels):")
            print(describe_arr("bbox_w", w))
            print(describe_arr("bbox_h", h))
            print(describe_arr("bbox_area", stats["bbox_area"]))

        if stats["bbox_rel_area"].size:
            print("\nBBox relative area (bbox_area / image_area):")
            print(describe_arr("bbox_rel_area", stats["bbox_rel_area"]))

        # save plots
        plot_class_counts(
            title=f"Class counts (annotations) ({split})",
            class_counts=stats["class_counts"],
            out_path=out_dir / f"class_counts_{split}.png",
        )
        plot_class_counts(
            title=f"Images per class ({split})",
            class_counts=stats["images_per_class"],
            out_path=out_dir / f"images_per_class_{split}.png",
        )
        plot_hist(
            title=f"Annotations per image ({split})",
            arr=api.astype(np.float32),
            out_path=out_dir / f"ann_per_image_{split}.png",
            bins=30,
            xlabel="annotations per image",
        )
        plot_hist(
            title=f"BBox area (pixels^2) ({split})",
            arr=stats["bbox_area"],
            out_path=out_dir / f"bbox_area_{split}.png",
            bins=50,
            xlabel="bbox area",
        )
        plot_hist(
            title=f"BBox relative area ({split})",
            arr=stats["bbox_rel_area"],
            out_path=out_dir / f"bbox_rel_area_{split}.png",
            bins=50,
            xlabel="bbox_area / image_area",
        )

        # NEW: example visualizations
        if img_dir_root is not None:
            # Try both conventions:
            # 1) img_dir_root/<split>
            # 2) img_dir_root (flat)
            cand1 = img_dir_root / split
            img_dir = cand1 if cand1.exists() else img_dir_root

            examples_dir = out_dir / f"examples_{split}"
            visualize_examples(coco, img_dir=img_dir, out_dir=examples_dir,
                               max_images=args.num_examples, seed=args.seed)

    # Save JSON report
    report_path = out_dir / "dataset_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print("\n✅ Rapor ve grafikler kaydedildi:")
    print(f"- {out_dir}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()

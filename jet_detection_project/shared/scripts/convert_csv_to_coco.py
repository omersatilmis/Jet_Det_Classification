#!/usr/bin/env python3
"""
CSV -> COCO dönüştürücü (jet tespiti dataseti için)

- Girdi:  labels_with_split.csv
- Çıktı:  instances_train.json, instances_validation.json, instances_test.json

CSV formatı:
    filename, width, height, class, xmin, ymin, xmax, ymax, split[, ext]
"""

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="labels_with_split.csv dosyasını COCO formatına çevir."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(REPO_ROOT / "archive" / "labels_with_split.csv"),
        help="labels_with_split.csv dosyasının yolu",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(REPO_ROOT / "archive" / "dataset"),
        help="Görsellerin bulunduğu klasör",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "coco_annotations"),
        help="COCO JSON dosyalarının yazılacağı klasör",
    )

    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=["F16", "F18", "F22", "F35"],
        help=(
            "Sadece bu sınıflar kullanılır (örn: --classes F16 F18 F22 F35). "
            "Boş bırakılırsa CSV'deki tüm sınıflar kullanılır."
        ),
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="Görüntü uzantısı (örn: .jpg). None ise dosya adından/CSV'den çıkarılır.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Görüntü dosyalarının varlığını kontrol et (yavaşlayabilir).",
    )
    return parser.parse_args()


def sanitize_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """BBox koordinatlarını düzelt (swap + clamp). Geçersiz ise None döndür."""
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))

    w = xmax - xmin
    h = ymax - ymin

    if w <= 1 or h <= 1:
        return None

    return float(xmin), float(ymin), float(w), float(h)


def resolve_filename(row, image_ext: Optional[str]) -> str:
    """
    Dosya adını belirler:
    - image_ext verilmişse onunla bitirir
    - Yoksa row['filename'] içinde uzantı varsa kullanır
    - Yoksa row['ext'] / row['extension'] varsa ekler
    - Hiçbiri yoksa .jpg ekler
    """
    raw = str(row["filename"]).strip()
    row_ext = None
    for key in ["ext", "extension"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            row_ext = row[key].strip()
            break

    if image_ext:
        base, ext = os.path.splitext(raw)
        return base + image_ext if base else raw + image_ext

    if "." in raw:
        return raw

    if row_ext:
        return raw + (row_ext if row_ext.startswith(".") else "." + row_ext)

    return raw + ".jpg"


def build_coco_for_split(
    df: pd.DataFrame,
    split_name: str,
    class_whitelist: Optional[List[str]],
    image_ext: Optional[str],
    images_dir: str,
    verify_images: bool = False,
) -> Optional[Dict]:
    df_split = df[df["split"] == split_name].copy()
    if df_split.empty:
        return None

    if class_whitelist is not None:
        df_split = df_split[df_split["class"].isin(class_whitelist)]
        if df_split.empty:
            return None

    df_split = df_split.reset_index(drop=True)

    if class_whitelist is not None:
        classes = [c for c in class_whitelist if c in df_split["class"].unique()]
    else:
        classes = sorted(df_split["class"].unique())

    if not classes:
        return None

    category_id_map = {name: i + 1 for i, name in enumerate(classes)}

    images: List[Dict] = []
    annotations: List[Dict] = []
    categories: List[Dict] = []

    for name in classes:
        categories.append(
            {"id": category_id_map[name], "name": name, "supercategory": "aircraft"}
        )

    image_id_map: Dict[str, int] = OrderedDict()
    ann_id = 1
    skipped_bbox = 0

    print(f"\n[{split_name}] Toplam satır: {len(df_split)}")

    for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
        file_name = resolve_filename(row, image_ext)

        img_w = int(row["width"])
        img_h = int(row["height"])
        cls_name = str(row["class"]).strip()

        if cls_name not in category_id_map:
            continue

        if file_name not in image_id_map:
            if verify_images:
                img_path = os.path.join(images_dir, file_name)
                if not os.path.exists(img_path):
                    continue

            img_id = len(image_id_map) + 1
            image_id_map[file_name] = img_id
            images.append(
                {"id": img_id, "file_name": file_name, "width": img_w, "height": img_h}
            )

        image_id = image_id_map[file_name]

        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        bbox = sanitize_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        if bbox is None:
            skipped_bbox += 1
            continue

        x, y, w, h = bbox
        area = float(w * h)

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(category_id_map[cls_name]),
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    if not annotations:
        return None

    if skipped_bbox:
        print(f"[WARN] {skipped_bbox} bbox filtrelendi (çok küçük/bozuk).")

    coco = {
        "info": {
            "description": f"Jet Aircraft Detection ({split_name})",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"CSV yükleniyor: {args.csv}")
    df = pd.read_csv(args.csv)

    required_cols = {
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "split",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV içinde eksik kolonlar var: {missing}")

    if args.classes is not None and len(args.classes) > 0:
        class_whitelist = [c.strip() for c in args.classes]
        print("Kullanılacak sınıflar (whitelist):", class_whitelist)
    else:
        class_whitelist = None
        print("Tüm sınıflar kullanılacak (whitelist yok).")

    splits = ["train", "validation", "test"]
    for split_name in splits:
        coco = build_coco_for_split(
            df=df,
            split_name=split_name,
            class_whitelist=class_whitelist,
            image_ext=args.ext,
            images_dir=args.images_dir,
            verify_images=args.verify_images,
        )

        if coco is None:
            print(f"[{split_name}] için kullanılabilir annotation bulunamadı, atlanıyor.")
            continue

        out_path = os.path.join(args.output_dir, f"instances_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(
            f"[{split_name}] -> {len(coco['images'])} görüntü, "
            f"{len(coco['annotations'])} annotation, "
            f"{len(coco['categories'])} sınıf yazıldı: {out_path}"
        )


if __name__ == "__main__":
    main()

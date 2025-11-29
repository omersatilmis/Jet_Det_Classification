import argparse
import json
import subprocess
import sys
from pathlib import Path
import re

import torch
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(description="Train jet detection model (Cascade R-CNN R50)")
    parser.add_argument("--config", type=str, default="mmdetection/configs/cascade_rcnn_convnext_tiny.py")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--prepare-coco", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Kaldığı yerden devam (auto checkpoint bulur)")
    parser.add_argument("--fresh", action="store_true", help="Sıfırdan başlat (resume kapalı)")
    parser.add_argument("--load-from", type=str, default=None, help="Sadece ağırlık yükle (epoch reset olur)")
    parser.add_argument("--evaluate", action="store_true", help="Only run evaluation (no training)")
    parser.add_argument("--tta", action="store_true", help="Evaluation sırasında test-time augmentation uygula")
    parser.add_argument("--seed", type=int, default=None, help="Tekrarlanabilirlik için global seed")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic backend ayarları")
    return parser.parse_args()


def _resolve_base_dataset(dataset_cfg):
    current = dataset_cfg
    while current is not None:
        nested = getattr(current, "dataset", None)
        if nested is None:
            return current
        current = nested
    return dataset_cfg


def apply_tta_cfg(cfg: Config):
    if not hasattr(cfg, "tta_model") or not hasattr(cfg, "tta_pipeline"):
        raise RuntimeError("Config içinde tta_model/tta_pipeline tanımlı değil.")

    cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    if hasattr(cfg, "test_dataloader") and hasattr(cfg.test_dataloader, "dataset"):
        test_ds = _resolve_base_dataset(cfg.test_dataloader.dataset)
        test_ds.pipeline = cfg.tta_pipeline
        test_ds.test_mode = True


def run_prepare_dataset(project_root: Path):
    prepare_script = project_root / "shared" / "scripts" / "prepare_jet_dataset.py"
    if not prepare_script.exists():
        print(f"[ERROR] prepare_jet_dataset.py bulunamadı: {prepare_script}")
        sys.exit(1)

    print(f"[INFO] COCO anotasyonları hazırlanıyor: {prepare_script}")
    cmd = [sys.executable, str(prepare_script)]
    subprocess.run(cmd, check=True)


def patch_cfg_paths(cfg: Config, project_root: Path):
    """
    Profesyonel yol yönetimi:
    - archive/dataset: Göreceli olarak proje dışındaki veri setini bulur.
    - coco_annotations: Proje içindeki JSON dosyalarını bulur.
    - work_dirs: Çıktıları düzenli bir klasöre yazar.
    """
    # 1. Klasörleri tanımla
    archive_dir = project_root.parent / "archive"
    images_dir = archive_dir / "dataset"
    ann_dir = project_root / "coco_annotations"
    
    # 2. Varlık denetimi
    if not images_dir.exists():
        print(f"[ERROR] Görsel klasörü bulunamadı: {images_dir}")
    if not ann_dir.exists():
        print(f"[ERROR] Anotasyon klasörü bulunamadı: {ann_dir}")

    # 3. Dataloader'ları tek tek gezerek tam yolları enjekte et
    for mode in ["train", "val", "test"]:
        dataloader = getattr(cfg, f"{mode}_dataloader", None)
        if not dataloader: continue

        ds = _resolve_base_dataset(dataloader.dataset)
        ds.data_root = "" # Tam yol kullanacağımız için root'u sıfırlıyoruz
        ds.data_prefix = dict(img=str(images_dir) + "/") # Görsel yolu
        
        # Anotasyon dosya adını path ile birleştir
        # (Config içinde sadece dosya adı kalmıştı)
        if mode == "train":
            ds.ann_file = str(ann_dir / "instances_train.json")
        elif mode == "val":
            ds.ann_file = str(ann_dir / "instances_validation.json")
        elif mode == "test":
            ds.ann_file = str(ann_dir / "instances_test.json")

    # 4. Evaluator (Değerlendirme) yollarını güncelle
    if hasattr(cfg, "val_evaluator"):
        cfg.val_evaluator.ann_file = str(ann_dir / "instances_validation.json")
    if hasattr(cfg, "test_evaluator"):
        cfg.test_evaluator.ann_file = str(ann_dir / "instances_test.json")

    print(f"[OK] Dinamik yol yamama tamamlandı. Resimler: {images_dir.name}")


def validate_dataset_consistency(cfg: Config):
    split_to_key = {
        "train": "train_dataloader",
        "val": "val_dataloader",
        "test": "test_dataloader",
    }

    expected_classes = None
    expected_num_classes = None

    train_ds = _resolve_base_dataset(getattr(cfg.train_dataloader, "dataset", None))
    if train_ds is not None:
        metainfo = getattr(train_ds, "metainfo", None)
        if metainfo is not None:
            expected_classes = tuple(metainfo.get("classes", ()))

    try:
        bbox_head = cfg.model.roi_head.bbox_head
        if isinstance(bbox_head, list) and bbox_head:
            expected_num_classes = int(bbox_head[0].num_classes)
    except Exception:
        expected_num_classes = None

    errors = []
    checked = []

    for split, key in split_to_key.items():
        dataloader = getattr(cfg, key, None)
        if dataloader is None:
            errors.append(f"{split}: dataloader bulunamadı ({key})")
            continue

        dataset = _resolve_base_dataset(getattr(dataloader, "dataset", None))
        if dataset is None:
            errors.append(f"{split}: dataset bulunamadı")
            continue

        data_prefix = getattr(dataset, "data_prefix", {}) or {}
        img_dir = Path(data_prefix.get("img", "")).expanduser()
        ann_file = Path(str(getattr(dataset, "ann_file", ""))).expanduser()

        if not img_dir.exists():
            errors.append(f"{split}: image klasörü bulunamadı -> {img_dir}")
        if not ann_file.exists():
            errors.append(f"{split}: annotation json bulunamadı -> {ann_file}")
            continue

        try:
            with ann_file.open("r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception as e:
            errors.append(f"{split}: json okunamadı -> {ann_file} ({e})")
            continue

        categories = coco.get("categories", [])
        annotations = coco.get("annotations", [])
        images = coco.get("images", [])

        if not isinstance(categories, list) or not categories:
            errors.append(f"{split}: categories boş/geçersiz -> {ann_file}")
            continue

        cat_names = [c.get("name") for c in categories if "name" in c]
        cat_ids = {c.get("id") for c in categories if "id" in c}
        ann_cat_ids = {a.get("category_id") for a in annotations if "category_id" in a}

        if expected_num_classes is not None and len(cat_names) != expected_num_classes:
            errors.append(
                f"{split}: class sayısı uyuşmuyor (json={len(cat_names)} model={expected_num_classes})"
            )

        if expected_classes:
            if set(cat_names) != set(expected_classes):
                errors.append(
                    f"{split}: class isimleri uyuşmuyor (json={sorted(set(cat_names))} cfg={sorted(set(expected_classes))})"
                )

        if ann_cat_ids and not ann_cat_ids.issubset(cat_ids):
            errors.append(f"{split}: annotation category_id değerleri categories ile uyumsuz")

        checked.append((split, len(images), len(annotations), len(cat_names), ann_file))

    if errors:
        msg = "\n".join(["[DATASET VALIDATION FAILED]"] + [f"- {e}" for e in errors])
        raise RuntimeError(msg)

    print("[OK] Dataset doğrulaması başarılı:")
    for split, image_count, ann_count, class_count, ann_file in checked:
        print(
            f"  - {split}: images={image_count}, anns={ann_count}, classes={class_count} | {ann_file.name}"
        )


def _read_last_checkpoint(work_dir: Path) -> Path | None:
    """
    MMEngine genelde work_dir içine 'last_checkpoint' dosyası yazar.
    İçinde 'epoch_xx.pth' gibi bir path olur.
    """
    p = work_dir / "last_checkpoint"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8").strip().strip('"').strip("'")
    if not txt:
        return None
    ckpt = Path(txt)
    # bazen sadece dosya adı yazar -> work_dir ile birleştir
    if not ckpt.is_absolute():
        ckpt = work_dir / ckpt
    return ckpt if ckpt.exists() else None


def _find_latest_epoch_ckpt(work_dir: Path) -> Path | None:
    """
    epoch_*.pth içinden en büyüğünü bulur.
    """
    candidates = list(work_dir.glob("epoch_*.pth"))
    if not candidates:
        return None

    def _epoch_num(p: Path) -> int:
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    candidates.sort(key=_epoch_num)
    best = candidates[-1]
    return best if best.exists() else None


def find_resume_checkpoint(work_dir: Path) -> Path | None:
    """
    Öncelik:
      1) last_checkpoint -> işaret ettiği ckpt
      2) latest.pth
      3) epoch_*.pth max
    """
    ckpt = _read_last_checkpoint(work_dir)
    if ckpt is not None:
        return ckpt

    ckpt = work_dir / "latest.pth"
    if ckpt.exists():
        return ckpt

    return _find_latest_epoch_ckpt(work_dir)


def peek_checkpoint_meta(ckpt_path: Path) -> dict:
    """
    Checkpoint içinden meta/epoch/iter okumaya çalışır.
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as e:
        return {"error": f"checkpoint okunamadı: {e}"}

    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    # MMEngine: meta içinde epoch, iter olabilir
    epoch = meta.get("epoch", None)
    it = meta.get("iter", None) or meta.get("iteration", None)
    return {
        "epoch": epoch,
        "iter": it,
        "meta_keys": list(meta.keys())[:20] if isinstance(meta, dict) else None,
    }


def main():
    args = parse_args()

    # Proje kökü: jet_detection_project
    project_root = Path(__file__).resolve().parents[2]  # .../codes/scripts -> parents[2] = project_root
    print(f"[INFO] Project root: {project_root}")

    if args.prepare_coco:
        run_prepare_dataset(project_root)

    cfg_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config dosyası bulunamadı: {cfg_path}")
        sys.exit(1)

    print(f"[INFO] Config yükleniyor: {cfg_path}")
    cfg = Config.fromfile(str(cfg_path))

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = str(project_root / args.work_dir) if not Path(args.work_dir).is_absolute() else args.work_dir
    else:
        cfg.work_dir = str(project_root / "work_dirs" / cfg_path.stem)

    work_dir = Path(cfg.work_dir)
    mkdir_or_exist(cfg.work_dir)
    print(f"[INFO] work_dir = {cfg.work_dir}")

    if args.seed is not None:
        cfg.randomness = dict(seed=int(args.seed), deterministic=bool(args.deterministic))
        print(f"[INFO] randomness -> seed={args.seed}, deterministic={args.deterministic}")

    # path patch
    patch_cfg_paths(cfg, project_root)
    validate_dataset_consistency(cfg)

    if args.evaluate and args.tta:
        apply_tta_cfg(cfg)
        print("[INFO] TTA evaluation aktif.")

    # RESUME / LOAD MANTIĞI
    if args.fresh:
        cfg.resume = False
        cfg.load_from = None
        if "resume_from" in cfg:
            cfg.resume_from = None
        print("[INFO] fresh=True => sıfırdan başlayacak.")
    else:
        # load_from verilmişse: initialize
        if args.load_from:
            load_from = Path(args.load_from).expanduser().resolve()
            if not load_from.exists():
                raise FileNotFoundError(f"--load-from bulunamadı: {load_from}")
            cfg.resume = False
            cfg.load_from = str(load_from)
            if "resume_from" in cfg:
                cfg.resume_from = None
            print(f"[INFO] load_from (init only): {load_from}")
            print("[INFO] (Not) load_from ile epoch sıfırlanır. 'resume' gibi devam etmez.")
        else:
            # resume default: auto if checkpoint exists
            ckpt = find_resume_checkpoint(work_dir)
            if ckpt is None:
                if args.resume:
                    print("[WARN] --resume verildi ama work_dir i??inde checkpoint bulunamad??. S??f??rdan ba?Ylayacak.")
                else:
                    print("[INFO] Auto-resume icin checkpoint bulunamadi. Sifirdan baslayacak.")
                cfg.resume = False
                cfg.load_from = None
                if "resume_from" in cfg:
                    cfg.resume_from = None
            else:
                meta = peek_checkpoint_meta(ckpt)
                print("=" * 70)
                if args.resume:
                    print(f"[RESUME] Bulunan checkpoint: {ckpt}")
                else:
                    print(f"[AUTO-RESUME] Bulunan checkpoint: {ckpt}")
                if "error" in meta:
                    print(f"[RESUME] Meta okunamad??: {meta['error']}")
                else:
                    print(f"[RESUME] Checkpoint meta -> epoch: {meta.get('epoch')} | iter: {meta.get('iter')}")
                print("=" * 70)

                # MMEngine: resume_from en net yol
                cfg.resume = True
                cfg.load_from = str(ckpt)   
                cfg.resume_from = str(ckpt) # <-- kritik

    runner = Runner.from_cfg(cfg)

    if args.evaluate:
        print("[INFO] Değerlendirme başlatılıyor...")
        runner.test()
        return

    print("[INFO] Eğitim başlatılıyor...")
    runner.train()


if __name__ == "__main__":
    main()

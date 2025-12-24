"""
Jet Detection – FPS & Model Benchmark
======================================
Her iki model (Cascade R-CNN + YOLOv8-S) için:
  - Inference FPS (warm-up sonrası)
  - Ortalama latency (ms/görsel)
  - Model parametre sayısı
  - GPU memory kullanımı
  - Sonuç tablosu (bitirme projesi raporu için)

Kullanım:
    python shared/scripts/benchmark_fps.py
    python shared/scripts/benchmark_fps.py --num-images 200
    python shared/scripts/benchmark_fps.py --only mmdet
    python shared/scripts/benchmark_fps.py --only mmyolo
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Model FPS Benchmark")
    p.add_argument("--only", choices=["mmdet", "mmyolo"], default=None,
                   help="Sadece belirtilen modeli test et")
    p.add_argument("--num-images", type=int, default=100,
                   help="Test edilecek görsel sayısı")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup inference sayısı (zamanlamaya dahil değil)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "mmdetection").exists() and (p / "coco_annotations").exists():
            return p
    raise RuntimeError("Proje kökü bulunamadı")


def find_best_checkpoint(work_dir: Path) -> Path | None:
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


def count_parameters(model) -> int:
    """Trainable + non-trainable parametreleri say."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_mb(device) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def collect_image_paths(ann_file: str, images_dir: str, max_images: int) -> list[str]:
    """COCO annotation dosyasından görsel yollarını çek."""
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    paths = []
    for img_info in coco.get("images", []):
        fname = img_info.get("file_name", "")
        full = os.path.join(images_dir, fname)
        if os.path.exists(full):
            paths.append(full)
        if len(paths) >= max_images:
            break

    return paths


def benchmark_model(model, image_paths, warmup_count, device):
    """Modeli benchmark et: FPS, latency, GPU memory."""
    from mmdet.apis import inference_detector

    # GPU memory reset
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Warmup
    print(f"  Warmup ({warmup_count} inference)...", end=" ", flush=True)
    for i in range(min(warmup_count, len(image_paths))):
        _ = inference_detector(model, image_paths[i % len(image_paths)])
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    print("OK")

    # GPU memory reset after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Benchmark
    latencies = []
    print(f"  Benchmark ({len(image_paths)} inference)...", flush=True)

    for idx, img_path in enumerate(image_paths):
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        _ = inference_detector(model, img_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

        if (idx + 1) % 50 == 0:
            avg_so_far = np.mean(latencies)
            print(f"    [{idx+1}/{len(image_paths)}] avg: {avg_so_far:.1f}ms "
                  f"({1000/avg_so_far:.1f} FPS)")

    gpu_mem = get_gpu_memory_mb(device)

    results = {
        "num_images": len(image_paths),
        "total_params": count_parameters(model),
        "trainable_params": count_trainable_parameters(model),
        "params_m": round(count_parameters(model) / 1e6, 2),
        "gpu_memory_mb": round(gpu_mem, 1),
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "latency_std_ms": round(float(np.std(latencies)), 2),
        "latency_median_ms": round(float(np.median(latencies)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "fps_mean": round(1000.0 / float(np.mean(latencies)), 1),
        "fps_median": round(1000.0 / float(np.median(latencies)), 1),
    }

    return results


def find_mmdet_checkpoint(project_root: Path) -> tuple[Path, Path] | None:
    config = project_root / "mmdetection" / "configs" / "cascade_rcnn_convnext_tiny.py"
    if not config.exists():
        return None

    # work_dir arama sırası
    for wd in [
        project_root / "work_dirs" / "mmdetection" / "cascade_rcnn_convnext_tiny_v2",
        project_root / "work_dirs" / "mmdetection" / "cascade_rcnn_convnext_tiny",
        project_root / "work_dirs" / "cascade_rcnn_convnext_tiny_v2",
        project_root / "work_dirs" / "cascade_rcnn_convnext_tiny",
        project_root / "work_dirs" / "cascade_rcnn_r50_tiny",
    ]:
        if wd.exists():
            ckpt = find_best_checkpoint(wd)
            if ckpt:
                return config, ckpt

    return None


def find_mmyolo_checkpoint(project_root: Path) -> tuple[Path, Path] | None:
    config = project_root / "mmyolo" / "configs" / "jet" / "yolov8_s_jet.py"
    if not config.exists():
        return None

    for wd in [
        project_root / "work_dirs" / "mmyolo" / "yolov8_s_jet_v2",
        project_root / "work_dirs" / "mmyolo" / "yolov8_s_jet",
        project_root / "work_dirs" / "yolov8_s_jet_v2",
        project_root / "work_dirs" / "yolov8_s_jet",
    ]:
        if wd.exists():
            ckpt = find_best_checkpoint(wd)
            if ckpt:
                return config, ckpt

    return None


def main():
    args = parse_args()
    project_root = find_project_root()
    print(f"[INFO] Project root: {project_root}")

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "outputs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Görsel yollarını topla
    ann_dir = project_root / "coco_annotations"
    ann_map = {"val": "instances_validation.json", "test": "instances_test.json"}
    ann_file = str(ann_dir / ann_map[args.split])

    archive_dir = project_root.parent / "archive" / "dataset"
    if not archive_dir.exists():
        archive_dir = project_root / "archive" / "dataset"

    image_paths = collect_image_paths(ann_file, str(archive_dir), args.num_images)
    print(f"[INFO] {len(image_paths)} görsel bulundu ({args.split} split)")

    if not image_paths:
        print("[ERROR] Test görselleri bulunamadı!")
        sys.exit(1)

    all_results = {}

    # ========== MMDetection Benchmark ==========
    if args.only is None or args.only == "mmdet":
        print("\n" + "=" * 70)
        print("BENCHMARK: Cascade R-CNN + ConvNeXt-Tiny (MMDetection)")
        print("=" * 70)

        mmdet_info = find_mmdet_checkpoint(project_root)
        if mmdet_info is None:
            print("  [SKIP] MMDetection checkpoint bulunamadı")
        else:
            config_path, ckpt_path = mmdet_info
            print(f"  Config : {config_path.name}")
            print(f"  Ckpt   : {ckpt_path.name}")

            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
            register_all_modules()

            model = init_detector(str(config_path), str(ckpt_path), device=args.device)
            results = benchmark_model(model, image_paths, args.warmup, args.device)
            results["model_name"] = "Cascade R-CNN + ConvNeXt-Tiny"
            results["checkpoint"] = ckpt_path.name
            all_results["mmdet"] = results

            print(f"\n  --- Sonuçlar ---")
            print(f"  Parametreler : {results['params_m']}M")
            print(f"  GPU Memory   : {results['gpu_memory_mb']} MB")
            print(f"  Latency      : {results['latency_mean_ms']:.1f}ms ± {results['latency_std_ms']:.1f}ms")
            print(f"  FPS          : {results['fps_mean']:.1f}")

            del model
            torch.cuda.empty_cache()

    # ========== MMYOLO Benchmark ==========
    if args.only is None or args.only == "mmyolo":
        print("\n" + "=" * 70)
        print("BENCHMARK: YOLOv8-S (MMYOLO)")
        print("=" * 70)

        yolo_info = find_mmyolo_checkpoint(project_root)
        if yolo_info is None:
            print("  [SKIP] MMYOLO checkpoint bulunamadı")
        else:
            config_path, ckpt_path = yolo_info
            print(f"  Config : {config_path.name}")
            print(f"  Ckpt   : {ckpt_path.name}")

            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
            register_all_modules()
            try:
                from mmyolo.utils import register_all_modules as register_yolo
                register_yolo()
            except ImportError:
                import mmyolo  # noqa: F401

            model = init_detector(str(config_path), str(ckpt_path), device=args.device)
            results = benchmark_model(model, image_paths, args.warmup, args.device)
            results["model_name"] = "YOLOv8-S"
            results["checkpoint"] = ckpt_path.name
            all_results["mmyolo"] = results

            print(f"\n  --- Sonuçlar ---")
            print(f"  Parametreler : {results['params_m']}M")
            print(f"  GPU Memory   : {results['gpu_memory_mb']} MB")
            print(f"  Latency      : {results['latency_mean_ms']:.1f}ms ± {results['latency_std_ms']:.1f}ms")
            print(f"  FPS          : {results['fps_mean']:.1f}")

            del model
            torch.cuda.empty_cache()

    # ========== ÖZET TABLO ==========
    print("\n" + "=" * 70)
    print("📊 KARŞILAŞTIRMA TABLOSU (Bitirme Projesi Raporu için)")
    print("=" * 70)

    header = f"  {'Model':<32} {'Params':>8} {'GPU MB':>8} {'Latency':>10} {'FPS':>8}"
    print(header)
    print("  " + "-" * 68)

    for key, r in all_results.items():
        print(f"  {r['model_name']:<32} {r['params_m']:>7}M {r['gpu_memory_mb']:>7.0f} "
              f"{r['latency_mean_ms']:>8.1f}ms {r['fps_mean']:>7.1f}")

    # JSON kaydet
    report_path = out_dir / "benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  → Sonuçlar: {report_path}")

    # Markdown tablo kaydet (kopyala-yapıştır için)
    md_path = out_dir / "benchmark_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Benchmark Sonuçları\n\n")
        f.write(f"| Model | Params (M) | GPU Mem (MB) | Latency (ms) | FPS |\n")
        f.write(f"|-------|-----------|-------------|-------------|-----|\n")
        for r in all_results.values():
            f.write(f"| {r['model_name']} | {r['params_m']} | {r['gpu_memory_mb']:.0f} | "
                    f"{r['latency_mean_ms']:.1f} ± {r['latency_std_ms']:.1f} | "
                    f"{r['fps_mean']:.1f} |\n")
        f.write(f"\n*{args.num_images} görsel, {args.split} split, {args.device}*\n")
    print(f"  → Markdown tablo: {md_path}")

    print("\n" + "=" * 70)
    print("BENCHMARK TAMAMLANDI ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()

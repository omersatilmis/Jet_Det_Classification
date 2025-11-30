import argparse
import time
from pathlib import Path
from typing import Optional, List

import cv2
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS


try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


def get_project_root() -> Path:
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path.cwd().resolve()

    for p in [start] + list(start.parents):
        has_mmdet = (p / "mmdetection" / "configs").exists()
        has_shared = (p / "shared" / "scripts").exists()
        has_workdirs = (p / "work_dirs").exists()
        if has_mmdet and has_shared and has_workdirs:
            return p
    raise RuntimeError("Proje kökü bulunamadı (mmdetection/configs + shared/scripts + work_dirs aranıyor)")


def pick_video_from_gui(initial_dir: Path) -> Path:
    try:
        from tkinter import Tk, filedialog
    except Exception as e:
        raise RuntimeError(
            "GUI video seçimi için tkinter gerekli ama import edilemedi. "
            "Çözüm: --video ile path ver veya tkinter kurulumunu kontrol et."
        ) from e

    if not initial_dir.exists():
        initial_dir = initial_dir.parent if initial_dir.parent.exists() else Path.cwd()

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askopenfilename(
        title="Bir video seç",
        initialdir=str(initial_dir),
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not selected:
        raise RuntimeError("Video seçilmedi (iptal edildi).")

    return Path(selected)


def resolve_path(project_root: Path, p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (project_root / pp)


def parse_args():
    ap = argparse.ArgumentParser(description="Jet Detection video inference (MMDetection 3.x)")
    ap.add_argument("--config", type=str, default="mmdetection/configs/cascade_rcnn_convnext_tiny.py")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--work-dir", type=str, default="work_dirs/cascade_rcnn_convnext_tiny")
    ap.add_argument("--ckpt-name", type=str, default="best_coco_bbox_mAP_epoch_21.pth")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.30)
    ap.add_argument("--resize-width", type=int, default=960, help="0 => resize kapalı")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out-name", type=str, default="pred_vis.mp4")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--use-sahi", action="store_true", help="Küçük nesneler için SAHI kullan")
    ap.add_argument("--slice-size", type=int, default=512, help="SAHI dilim boyutu")
    ap.add_argument("--frame-skip", type=int, default=1, help="N karede bir işle (FPS artırır)")
    return ap.parse_args()


def main():
    args = parse_args()
    PROJECT_ROOT = get_project_root()

    CONFIG_PATH = resolve_path(PROJECT_ROOT, args.config)
    if CONFIG_PATH is None:
        raise RuntimeError("Config yolu çözümlenemedi.")

    if args.checkpoint:
        CKPT_PATH = resolve_path(PROJECT_ROOT, args.checkpoint)
        if CKPT_PATH is None:
            raise RuntimeError("Checkpoint yolu çözümlenemedi.")
    else:
        work_dir = resolve_path(PROJECT_ROOT, args.work_dir)
        if work_dir is None:
            raise RuntimeError("work_dir yolu çözümlenemedi.")

        ckpt_candidates: List[Path] = [
            work_dir / args.ckpt_name,
            work_dir / "latest.pth",
        ]
        CKPT_PATH = next((p for p in ckpt_candidates if p.exists()), None)

        if CKPT_PATH is None and work_dir.exists():
            epoch_ckpts = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if epoch_ckpts:
                CKPT_PATH = epoch_ckpts[0]

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config yok: {CONFIG_PATH}")
    if CKPT_PATH is None or not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint yok: {CKPT_PATH}")

    VIDEOS_DIR = PROJECT_ROOT / "testing" / "video_input"
    DEFAULT_VIDEO = VIDEOS_DIR / "f35_footage.mp4"

    video_path = resolve_path(PROJECT_ROOT, args.video)

    if video_path is None:
        if args.no_gui:
            video_path = DEFAULT_VIDEO
        else:
            video_path = pick_video_from_gui(VIDEOS_DIR)

    if not video_path.exists():
        raise FileNotFoundError(f"Video yok: {video_path}")

    print("-" * 70)
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("CONFIG_PATH  :", CONFIG_PATH)
    print("CKPT_PATH    :", CKPT_PATH)
    print("VIDEO_PATH   :", video_path)
    print("DEVICE       :", args.device)
    print("SCORE_THR    :", args.score_thr)
    print("USE_SAHI     :", args.use_sahi and SAHI_AVAILABLE)
    print("-" * 70)

    if args.use_sahi and not SAHI_AVAILABLE:
        print("[WARN] SAHI yüklü değil, normal inference yapılacak.")

    model = None
    sahi_model = None

    if args.use_sahi and SAHI_AVAILABLE:
        sahi_model = AutoDetectionModel.from_model_type(
            model_type="mmdet",
            model_path=str(CKPT_PATH),
            config_path=str(CONFIG_PATH),
            device=args.device,
            confidence_threshold=args.score_thr,
        )
        # Visualizer için hala bir model nesnesi gerekebilir (metainfo için)
        model = init_detector(str(CONFIG_PATH), str(CKPT_PATH), device=args.device)
    else:
        model = init_detector(str(CONFIG_PATH), str(CKPT_PATH), device=args.device)

    print("[OK] Model loaded.")

    visualizer = VISUALIZERS.build(
        dict(type="DetLocalVisualizer", name="vis", line_width=3, alpha=0.8)
    )
    visualizer.dataset_meta = model.dataset_meta

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    # Pencereyi ÖNCEDEN yarat
    win_name = "Jet Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    writer = None
    out_path = None

    prev_t = time.time()
    fps_smooth = None
    frame_count = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_count += 1
        if args.frame_skip > 1 and (frame_count % args.frame_skip != 0):
            # Bu kareyi işleme ama eğer kaydediyorsak son sonucu veya boş kareyi yaz
            if writer is not None and 'out_bgr' in locals():
                writer.write(out_bgr)
            if 'out_bgr' in locals():
                cv2.imshow(win_name, out_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            continue

        if args.resize_width and args.resize_width > 0:
            h, w = frame_bgr.shape[:2]
            new_w = args.resize_width
            new_h = int(h * (new_w / w))
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Inference
        if sahi_model:
            # SAHI RGB bekler ve kendi içinde slice yapar
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            sahi_result = get_sliced_prediction(
                frame_rgb,
                sahi_model,
                slice_height=args.slice_size,
                slice_width=args.slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            # Şimdilik standart inference'ı fallback olarak kullanıyoruz
            # İleride sahi_result -> DetDataSample dönüşümü eklenebilir
            result = inference_detector(model, frame_bgr)
        else:
            result = inference_detector(model, frame_bgr)

        # Visualizer RGB istiyor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        visualizer.add_datasample(
            name="pred",
            image=frame_rgb,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=args.score_thr,
        )
        out_rgb = visualizer.get_image()
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_t))
        prev_t = now
        fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)

        cv2.putText(
            out_bgr,
            f"FPS: {fps_smooth:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        if args.save and writer is None:
            out_dir = PROJECT_ROOT / "testing" / "predicted"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / args.out_name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_h, out_w = out_bgr.shape[:2]

            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if not src_fps or src_fps <= 1:
                src_fps = 25.0

            writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h))
            print(f"[INFO] Output video: {out_path}")

        if writer is not None:
            writer.write(out_bgr)

        cv2.imshow(win_name, out_bgr)

        # Pencere kapandıysa çık (BU KONTROL DOĞRU YERDE: imshow SONRASI)
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"✅ Kaydedildi: {out_path}")

    cv2.destroyAllWindows()
    print("[OK] Bitti.")


if __name__ == "__main__":
    main()

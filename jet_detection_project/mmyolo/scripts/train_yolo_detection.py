import argparse
import sys
from pathlib import Path

from mmengine.config import Config, ConfigDict
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

try:
    from mmyolo.utils import is_metainfo_lower
except ImportError:
    is_metainfo_lower = None

try:
    from mmdet.utils import setup_cache_size_limit_of_dynamo
except ImportError:
    setup_cache_size_limit_of_dynamo = None


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train YOLO model with project defaults (MMYOLO).")
    parser.add_argument("--config", default=str(project_root / "mmyolo" / "configs" / "jet" / "yolov8_s_jet.py"))
    parser.add_argument("--work-dir", default=str(project_root / "work_dirs" / "mmyolo" / "yolov8_s_jet_v2"))
    parser.add_argument("--resume", action="store_true", help="Kaldığı yerden devam")
    args, _ = parser.parse_known_args()
    return args


def _resolve_base_dataset(dataset_cfg):
    current = dataset_cfg
    while current is not None:
        nested = getattr(current, "dataset", None)
        if nested is None:
            return current
        current = nested
    return dataset_cfg


def patch_cfg_paths(cfg: Config, project_root: Path):
    """Dynamically routes dataset configs directly to the actual dataset location."""
    archive_dir = project_root.parent / "archive"
    images_dir = archive_dir / "dataset"
    ann_dir = project_root / "coco_annotations"

    for mode in ["train", "val", "test"]:
        dataloader = getattr(cfg, f"{mode}_dataloader", None)
        if not dataloader: continue
        
        ds = _resolve_base_dataset(dataloader.dataset)
        ds.data_root = ""
        ds.data_prefix = dict(img=str(images_dir) + "/")

        if mode == "train":
            ds.ann_file = str(ann_dir / "instances_train.json")
        elif mode == "val":
            ds.ann_file = str(ann_dir / "instances_validation.json")
        elif mode == "test":
            ds.ann_file = str(ann_dir / "instances_test.json")

    if hasattr(cfg, "val_evaluator"):
        cfg.val_evaluator.ann_file = str(ann_dir / "instances_validation.json")
    if hasattr(cfg, "test_evaluator"):
        cfg.test_evaluator.ann_file = str(ann_dir / "instances_test.json")
        
    print(f"[OK] MMYOLO dataset paths patched dynamically. Images: {images_dir}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    # Initialize optimizations
    if setup_cache_size_limit_of_dynamo is not None:
        setup_cache_size_limit_of_dynamo()

    # Load configuration
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    if args.resume:
        cfg.resume = True

    # Critical path fix for datasets
    patch_cfg_paths(cfg, project_root)

    # MMYOLO specific startup tasks
    if is_metainfo_lower is not None:
        is_metainfo_lower(cfg)

    # Instantiate runner and start
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
        
    runner.train()

if __name__ == "__main__":
    main()

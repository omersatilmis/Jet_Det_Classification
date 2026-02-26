import sys
import argparse
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Import the existing COCO builder from the shared script
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "shared" / "scripts"))
try:
    from convert_csv_to_coco import build_coco_for_split
except ImportError:
    print("[ERROR] Cannot import build_coco_for_split from convert_csv_to_coco.py")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Create K-Fold COCO datasets from CSV.")
    parser.add_argument("--csv", default=r"C:\Users\omerf\Desktop\archive\labels_with_k_fold.csv")
    parser.add_argument("--images-dir", default=r"C:\Users\omerf\Desktop\archive\dataset")
    parser.add_argument("--out-dir", default=str(repo_root / "coco_annotations"))
    parser.add_argument("-k", "--n-splits", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    return parser.parse_args()


def get_dominant_class_per_image(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by filename and finds the most frequent class per image.
    This is required for StratifiedKFold which assumes 1 label per sample (image).
    """
    # Create a dataframe with unique filenames and their most frequent class
    dominant_classes = (
        df.groupby('filename')['class']
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    return dominant_classes


def main():
    args = parse_args()
    
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Isolate and process 'test' split if it exists
    test_df = df[df["split"] == "test"].copy()
    if not test_df.empty:
        print(f"[INFO] Found {len(test_df)} annotations for 'test' split. Generating instances_test.json...")
        test_coco = build_coco_for_split(
            df=df, split_name="test", class_whitelist=None, 
            image_ext=None, images_dir=args.images_dir
        )
        if test_coco:
            with open(out_dir / "instances_test.json", "w", encoding="utf-8") as f:
                json.dump(test_coco, f)
            print("[OK] instances_test.json created.")
    
    # 2. Extract the unassigned pool (anything that is NOT test)
    pool_df = df[df["split"] != "test"].copy()
    if pool_df.empty:
        print("[ERROR] No data available for K-Fold splitting. Make sure CSV has rows where split != 'test'.")
        return
        
    print(f"\n[INFO] Starting {args.n_splits}-Fold Stratified Splitting on {len(pool_df)} annotations...")
    
    # StratifiedKFold needs image-level labels, not instance-level.
    # We find the dominant class for each image.
    img_df = get_dominant_class_per_image(pool_df)
    
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    fold_idx = 0
    for train_idx, val_idx in skf.split(img_df['filename'], img_df['class']):
        print(f"\n--- Generating Fold {fold_idx} ---")
        
        # Get filenames for this fold
        train_filenames = img_df.iloc[train_idx]['filename'].values
        val_filenames = img_df.iloc[val_idx]['filename'].values
        
        # Create DataFrames for this fold
        fold_train_df = pool_df[pool_df['filename'].isin(train_filenames)].copy()
        fold_val_df = pool_df[pool_df['filename'].isin(val_filenames)].copy()
        
        # Override the "split" column temporarily so build_coco_for_split can use it
        fold_train_df['split'] = 'train'
        fold_val_df['split'] = 'validation'
        
        # Combine them back to feed into the builder
        fold_df = pd.concat([fold_train_df, fold_val_df], ignore_index=True)
        
        # Build Train COCO
        train_coco = build_coco_for_split(
            df=fold_df, split_name="train", class_whitelist=None, 
            image_ext=None, images_dir=args.images_dir
        )
        
        # Build Val COCO
        val_coco = build_coco_for_split(
            df=fold_df, split_name="validation", class_whitelist=None, 
            image_ext=None, images_dir=args.images_dir
        )
        
        # Save to JSON
        train_out = out_dir / f"fold_{fold_idx}_train.json"
        val_out = out_dir / f"fold_{fold_idx}_val.json"
        
        with open(train_out, "w", encoding="utf-8") as f:
            json.dump(train_coco, f)
            
        with open(val_out, "w", encoding="utf-8") as f:
            json.dump(val_coco, f)
            
        print(f"[OK] Fold {fold_idx} saved: {train_out.name}, {val_out.name}")
        fold_idx += 1
        
    print("\n[SUCCESS] All folds processed. You can now train using --fold X.")


if __name__ == "__main__":
    main()

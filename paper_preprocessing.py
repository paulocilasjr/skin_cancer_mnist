#!/usr/bin/env python
import os
import shutil
import zipfile
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SAMPLES_PER_CLASS = 100
IMG_SIZE_220     = (220, 220)
IMG_SIZE_96      = (96, 96)
OUTPUT_DIR_1     = "./processed_data"          # leak-possible (paper logic)
OUTPUT_DIR_2     = "./processed_data_no_leak"  # leak-free (paper logic + no leakage)

# -----------------------------------------------------------------------------
# Utility: Load and sort metadata
# -----------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "dx" in df.columns:
        df = df.rename(columns={"dx": "label"})
    return df.sort_values(by=["image_id", "label"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Sampling for leak-possible (image-level stratified)
# -----------------------------------------------------------------------------
def sample_balanced_leak(df):
    sampled = []
    for label in sorted(df["label"].unique()):
        c = df[df["label"] == label]
        n = min(SAMPLES_PER_CLASS, len(c))
        sampled.append(c.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True)

# -----------------------------------------------------------------------------
# Sampling for no-leak (lesion-aware with fallback)
# -----------------------------------------------------------------------------
def sample_balanced_no_leak(df):
    sampled = []
    rng = np.random.default_rng(42)
    for label in sorted(df["label"].unique()):
        c = df[df["label"] == label]
        lids = c["lesion_id"].unique()
        rng.shuffle(lids)
        picks = []
        # 1 image per lesion
        for lid in lids:
            picks.append(c[c["lesion_id"] == lid].sample(n=1, random_state=42))
            if len(picks) >= SAMPLES_PER_CLASS:
                break
        picked = pd.concat(picks, ignore_index=True)
        # top-up if needed
        if len(picked) < SAMPLES_PER_CLASS:
            pool = c[~c["image_id"].isin(picked["image_id"])]
            if not pool.empty:
                need = SAMPLES_PER_CLASS - len(picked)
                picked = pd.concat([
                    picked,
                    pool.sample(n=min(need, len(pool)), random_state=42)
                ], ignore_index=True)
        sampled.append(picked)
    return pd.concat(sampled, ignore_index=True)

# -----------------------------------------------------------------------------
# Lesion-aware 80/20 split for no-leak
# -----------------------------------------------------------------------------
def split_by_lesion_80_20(df):
    train_parts, test_parts = [], []
    for label in sorted(df["label"].unique()):
        c = df[df["label"] == label]
        lids = c["lesion_id"].unique()
        if len(lids) < 2:
            train_parts.append(c)
        else:
            tr_lids, te_lids = train_test_split(lids, test_size=0.2, random_state=42)
            train_parts.append(c[c["lesion_id"].isin(tr_lids)])
            test_parts.append(c[c["lesion_id"].isin(te_lids)])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts,  ignore_index=True)
    )

# -----------------------------------------------------------------------------
# Feature extraction for traditional ML
# -----------------------------------------------------------------------------
def extract_features(img: Image.Image) -> dict:
    f = {}
    img220 = img.resize(IMG_SIZE_220, Image.Resampling.LANCZOS)
    arr = np.array(img220)
    # Color histogram (16 bins each channel)
    for i, col in enumerate("rgb"):
        hist = cv2.calcHist([arr], [i], None, [16], [0, 256]).flatten()
        for j, val in enumerate(hist):
            f[f"{col}_hist_{j}"] = float(val)
    # Grayscale for texture & shape
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Haralick (GLCM)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        f[f"haralick_{prop}"] = float(graycoprops(glcm, prop)[0, 0])
    # Hu Moments
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    for idx, val in enumerate(hu):
        f[f"hu_moment_{idx}"] = float(val)
    return f

# -----------------------------------------------------------------------------
# Save, augment, extract for one split
# -----------------------------------------------------------------------------
def save_split(df_split, outdir, split_id, image_dir, do_numeric):
    dir220 = os.path.join(outdir, "images_220")
    dir96  = os.path.join(outdir, "images_96")
    os.makedirs(dir220, exist_ok=True)
    os.makedirs(dir96,  exist_ok=True)

    metadata, features = [], []

    for _, row in df_split.iterrows():
        iid, lbl = row["image_id"], row["label"]
        src = os.path.join(image_dir, f"{iid}.jpg")
        try:
            im = Image.open(src).convert("RGB")
            for tag, img in [("orig", im), ("flip", im.transpose(Image.FLIP_LEFT_RIGHT))]:
                fn = f"{iid}_{tag}.jpg"
                # 220x220
                img.resize(IMG_SIZE_220, Image.Resampling.LANCZOS).save(
                    os.path.join(dir220, fn), "JPEG", quality=95
                )
                # 96x96
                img.resize(IMG_SIZE_96, Image.Resampling.LANCZOS).save(
                    os.path.join(dir96, fn), "JPEG", quality=95
                )
                metadata.append({"image_path": fn, "label": lbl, "split": split_id})
            if do_numeric:
                feat = extract_features(im)
                feat.update({"image_path": f"{iid}_orig.jpg", "label": lbl, "split": split_id})
                features.append(feat)
        except Exception as e:
            print(f"Skipping {src}: {e}")

    return metadata, features

# -----------------------------------------------------------------------------
# Leakage check & summary
# -----------------------------------------------------------------------------
def verify_leakage(tr, va, te):
    ids_tr, ids_va, ids_te = set(tr["lesion_id"]), set(va["lesion_id"]), set(te["lesion_id"])
    for name, shared in [("TRAIN/VAL", ids_tr & ids_va),
                         ("TRAIN/TEST", ids_tr & ids_te),
                         ("VAL/TEST",   ids_va & ids_te)]:
        if shared:
            print(f"❌ Leakage detected {name}: {len(shared)} lesions")
    if not (ids_tr & ids_va or ids_tr & ids_te or ids_va & ids_te):
        print("✅ No lesion_id overlap between splits")

def summarize_split(df, label):
    print(f"\n{label} set: {len(df)} samples")
    if not df.empty:
        for cls, cnt in df["label"].value_counts().items():
            print(f"  {cls}: {cnt}")

# -----------------------------------------------------------------------------
# Create full dataset
# -----------------------------------------------------------------------------
def create_dataset(df, outdir, image_dir, leak_mode):
    os.makedirs(outdir, exist_ok=True)

    if leak_mode:
        # PAPER leak logic: 100/class → 80/20 image-level
        bal = sample_balanced_leak(df)
        tr, te = train_test_split(bal, test_size=0.2, random_state=42, stratify=bal["label"])
        va = pd.DataFrame(columns=bal.columns)
    else:
        # PAPER no-leak logic: 100/class lesion-aware → 80/20 lesion-level
        bal = sample_balanced_no_leak(df)
        tr, te = split_by_lesion_80_20(bal)
        va = pd.DataFrame(columns=bal.columns)

    print(f"\n===== Dataset: {'leak' if leak_mode else 'no_leak'} =====")
    verify_leakage(tr, va, te)
    summarize_split(tr, "TRAIN")
    summarize_split(va, "VAL")
    summarize_split(te, "TEST")

    meta_all, feat_all = [], []
    for split_df, sid in [(tr, 0), (va, 1), (te, 2)]:
        m, f = save_split(split_df, outdir, sid, image_dir, do_numeric=True)
        meta_all += m
        feat_all += f

    pd.DataFrame(meta_all).to_csv(os.path.join(outdir, "image_metadata.csv"), index=False)
    pd.DataFrame(feat_all).to_csv(os.path.join(outdir, "features_ludwig.csv"),  index=False)

    # -----------------------------------------------------------------------------
    # finally: zip up **all** of the images_96 and images_220 folders at once
    # -----------------------------------------------------------------------------
    for folder, zipname in [("images_96", "selected_images_96.zip"),
                            ("images_220", "selected_images_220.zip")]:
        folder_path = os.path.join(outdir, folder)
        zip_path    = os.path.join(outdir, zipname)
        with zipfile.ZipFile(zip_path, "w") as zf:
            for fname in os.listdir(folder_path):
                zf.write(os.path.join(folder_path, fname), arcname=fname)
        shutil.rmtree(folder_path)

    print(f"\n✅ Finished {outdir}:")
    print(f"  metadata rows: {len(meta_all)}")
    print(f"  feature rows:  {len(feat_all)}")
    print(f"  archives: selected_images_96.zip, selected_images_220.zip")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="HAM10000 paper preprocessing")
    p.add_argument("--csv",    required=True, help="Path to metadata CSV")
    p.add_argument("--images", required=True, help="Path to folder of original images")
    p.add_argument("--dataset", choices=["leak","no_leak"],
                   help="Which dataset to build (omit => both)")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    df = load_data(args.csv)

    if args.dataset == "leak":
        create_dataset(df, OUTPUT_DIR_1, args.images, leak_mode=True)
    elif args.dataset == "no_leak":
        create_dataset(df, OUTPUT_DIR_2, args.images, leak_mode=False)
    else:
        create_dataset(df, OUTPUT_DIR_1, args.images, leak_mode=True)
        create_dataset(df, OUTPUT_DIR_2, args.images, leak_mode=False)

if __name__ == "__main__":
    main()

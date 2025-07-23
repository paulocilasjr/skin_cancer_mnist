#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import zipfile
import argparse
import shutil
import cv2
from skimage.feature import graycomatrix, graycoprops

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SAMPLES_PER_CLASS = 100
IMG_SIZE_220 = (220, 220)
IMG_SIZE_96  = (96,  96)
OUTPUT_DIR_1 = "./processed_data"          # leak-possible (paper logic)
OUTPUT_DIR_2 = "./processed_data_no_leak"  # leak-free (paper logic + no leakage)

# -----------------------------------------------------------------------------
# Load Metadata
# -----------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "dx" in df.columns:
        df = df.rename(columns={"dx": "label"})
    return df.sort_values(by=["image_id", "label"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Sampling Helpers
# -----------------------------------------------------------------------------
def sample_balanced_leak(df):
    sampled = []
    for label in sorted(df['label'].unique()):
        c = df[df['label']==label]
        n = min(SAMPLES_PER_CLASS, len(c))
        sampled.append(c.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True)

def sample_balanced_no_leak(df):
    sampled, rng = [], np.random.default_rng(42)
    for label in sorted(df['label'].unique()):
        c = df[df['label']==label]
        lids = c['lesion_id'].unique()
        rng.shuffle(lids)
        picks = []
        # one per lesion
        for lid in lids:
            picks.append(c[c['lesion_id']==lid].sample(n=1, random_state=42))
            if len(picks)>=SAMPLES_PER_CLASS:
                break
        picked = pd.concat(picks, ignore_index=True)
        # top‑up if needed
        if len(picked)<SAMPLES_PER_CLASS:
            pool = c[~c['image_id'].isin(picked['image_id'])]
            if not pool.empty:
                need = SAMPLES_PER_CLASS - len(picked)
                picked = pd.concat([picked, pool.sample(n=need, random_state=42)], ignore_index=True)
        sampled.append(picked)
    return pd.concat(sampled, ignore_index=True)

# -----------------------------------------------------------------------------
# Lesion‑aware splitting (no_leak path) – 80/20 train/test
# -----------------------------------------------------------------------------
def split_by_lesion_80_20(df):
    train_parts, test_parts = [], []
    for label in sorted(df['label'].unique()):
        c = df[df['label']==label]
        lids = c['lesion_id'].unique()
        if len(lids)<2:
            # too few lesions → all to train
            train_parts.append(c)
            continue
        tr_lids, te_lids = train_test_split(lids, test_size=0.2, random_state=42)
        train_parts.append(c[c['lesion_id'].isin(tr_lids)])
        test_parts.append( c[c['lesion_id'].isin(te_lids)] )
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts,  ignore_index=True),
    )

# -----------------------------------------------------------------------------
# Numeric Feature Extraction (for classical ML)
# -----------------------------------------------------------------------------
def extract_features(img):
    f = {}
    img220 = img.resize(IMG_SIZE_220, Image.Resampling.LANCZOS)
    arr = np.array(img220)
    # Color hist (16 bins)
    for i,c in enumerate(['r','g','b']):
        h = cv2.calcHist([arr],[i],None,[16],[0,256]).flatten()
        for j,val in enumerate(h):
            f[f'{c}_hist_{j}'] = float(val)
    # Grayscale → Haralick
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    for p in ['contrast','dissimilarity','homogeneity','energy','correlation']:
        f[f'haralick_{p}'] = float(graycoprops(glcm,p)[0,0])
    # Hu moments
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m).flatten()
    for i,val in enumerate(hu):
        f[f'hu_moment_{i}'] = float(val)
    return f

# -----------------------------------------------------------------------------
# Save & Augment & Extract for one split
# -----------------------------------------------------------------------------
def save_split(df, outdir, split_id, image_dir, do_numeric):
    dir220 = os.path.join(outdir, "images_220")
    dir96  = os.path.join(outdir, "images_96")
    os.makedirs(dir220, exist_ok=True)
    os.makedirs(dir96,  exist_ok=True)
    meta, feats = [], []

    for _,r in df.iterrows():
        iid, lbl = r['image_id'], r['label']
        src = os.path.join(image_dir, f"{iid}.jpg")
        try:
            im = Image.open(src).convert('RGB')
            for tag, img in [('orig',im), ('flip', im.transpose(Image.FLIP_LEFT_RIGHT))]:
                fn = f"{iid}_{tag}.jpg"
                p220 = os.path.join(dir220, fn)
                img.resize(IMG_SIZE_220, Image.Resampling.LANCZOS).save(p220, 'JPEG', quality=95)
                # 96x96
                img.resize(IMG_SIZE_96, Image.Resampling.LANCZOS).save(os.path.join(dir96, fn), 'JPEG', quality=95)
                meta.append({"image_path": fn, "label": lbl, "split": split_id})
            if do_numeric:
                fdict = extract_features(im)
                fdict.update({"image_path": f"{iid}_orig.jpg", "label": lbl, "split": split_id})
                feats.append(fdict)
        except Exception as e:
            print(f"Skip {src}: {e}")

    # zip and cleanup
    for d,z in [(dir220, "selected_images_220.zip"), (dir96, "selected_images_96.zip")]:
        zp = os.path.join(outdir, z)
        with zipfile.ZipFile(zp,'w') as zf:
            for f in os.listdir(d):
                zf.write(os.path.join(d,f), arcname=f)
        shutil.rmtree(d)

    return meta, feats

# -----------------------------------------------------------------------------
# Dataset creation
# -----------------------------------------------------------------------------
def create_dataset(df, outdir, image_dir, leak_mode):
    os.makedirs(outdir, exist_ok=True)

    if leak_mode:
        # PAPER leak logic: 100/class → 80/20 split → augment
        bal = sample_balanced_leak(df)
        tr, te = train_test_split(bal, test_size=0.2, random_state=42, stratify=bal['label'])
        splits = [(tr,0,True), (te,2,True)]
    else:
        # NO‑LEAK paper logic: 100/class lesion-aware → 80/20 lesion split → augment
        bal = sample_balanced_no_leak(df)
        tr, te = split_by_lesion_80_20(bal)
        splits = [(tr,0,True), (te,2,True)]

    meta_all, feat_all = [], []
    for df_s, sid, do_num in splits:
        m,f = save_split(df_s, outdir, sid, image_dir, do_num)
        meta_all += m
        feat_all += f

    pd.DataFrame(meta_all).to_csv(os.path.join(outdir, "image_metadata.csv"), index=False)
    pd.DataFrame(feat_all).to_csv(os.path.join(outdir, "features_ludwig.csv"), index=False)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="HAM10000 preprocessing (paper logic)")
    p.add_argument('--csv',    required=True, help="Metadata CSV")
    p.add_argument('--images', required=True, help="Folder with original .jpg images")
    p.add_argument('--dataset', choices=['leak','no_leak'], help="Which dataset to build")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    df = load_data(args.csv)

    if args.dataset == 'leak':
        create_dataset(df, OUTPUT_DIR_1, args.images, leak_mode=True)
    elif args.dataset == 'no_leak':
        create_dataset(df, OUTPUT_DIR_2, args.images, leak_mode=False)
    else:
        create_dataset(df, OUTPUT_DIR_1, args.images, leak_mode=True)
        create_dataset(df, OUTPUT_DIR_2, args.images, leak_mode=False)

if __name__ == "__main__":
    main()

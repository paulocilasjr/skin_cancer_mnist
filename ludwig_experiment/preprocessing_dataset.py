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
IMAGE_SIZE = (220, 220)
OUTPUT_DIR_1 = "./processed_data"
OUTPUT_DIR_2 = "./processed_data_no_leak"

# -----------------------------------------------------------------------------
# Load Metadata
# -----------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "dx" in df.columns:
        df = df.rename(columns={"dx": "label"})
    return df.sort_values(by=["image_id", "label"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------------------------
def sample_balanced_leak(df):
    sampled = []
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        n = min(SAMPLES_PER_CLASS, len(class_df))
        sampled.append(class_df.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True)

def sample_balanced_no_leak(df):
    sampled = []
    rng = np.random.default_rng(42)
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        grouped = class_df.groupby("lesion_id")
        lesion_ids = grouped.size().index.to_numpy()
        rng.shuffle(lesion_ids)

        picked_rows = []
        for lid in lesion_ids:
            one = class_df[class_df['lesion_id'] == lid].sample(n=1, random_state=42)
            picked_rows.append(one)
            if len(picked_rows) >= SAMPLES_PER_CLASS:
                break

        picked_df = pd.concat(picked_rows, ignore_index=True)
        if len(picked_df) < SAMPLES_PER_CLASS:
            need = SAMPLES_PER_CLASS - len(picked_df)
            pool = class_df[~class_df['image_id'].isin(picked_df['image_id'])]
            if len(pool) > 0:
                picked_df = pd.concat([picked_df, pool.sample(n=min(need, len(pool)), random_state=42)], ignore_index=True)
        sampled.append(picked_df)
    return pd.concat(sampled, ignore_index=True)

# -----------------------------------------------------------------------------
# Stratified Lesion-Level Splitting
# -----------------------------------------------------------------------------
def split_by_lesion_stratified(df):
    train_parts, val_parts, test_parts = [], [], []
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        lesion_ids = class_df['lesion_id'].unique()
        if len(lesion_ids) < 3:
            train_parts.append(class_df)
            continue
        train_ids, temp_ids = train_test_split(lesion_ids, test_size=0.3, random_state=42)
        if len(temp_ids) == 1:
            val_ids, test_ids = temp_ids, []
        else:
            val_ids, test_ids = train_test_split(temp_ids, test_size=2/3, random_state=42)

        train_parts.append(class_df[class_df['lesion_id'].isin(train_ids)])
        val_parts.append(class_df[class_df['lesion_id'].isin(val_ids)])
        test_parts.append(class_df[class_df['lesion_id'].isin(test_ids)])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )

# -----------------------------------------------------------------------------
# Feature Extraction
# -----------------------------------------------------------------------------
def extract_features(img_pil):
    features = {}
    img_resized = img_pil.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_np = np.array(img_resized)

    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img_np], [i], None, [16], [0, 256]).flatten()
        for j, val in enumerate(hist):
            features[f'{color}_hist_{j}'] = float(val)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features[f'haralick_{prop}'] = float(graycoprops(glcm, prop)[0, 0])

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    for i, val in enumerate(hu):
        features[f'hu_moment_{i}'] = float(val)

    return features

# -----------------------------------------------------------------------------
# Image Saving, Feature Extraction, and Metadata Generation
# -----------------------------------------------------------------------------
def save_and_augment(df, output_dir, split_value, metadata, image_dir):
    images_out = os.path.join(output_dir, "images")
    os.makedirs(images_out, exist_ok=True)
    features_list = []

    for _, row in df.iterrows():
        image_filename = f"{row['image_id']}.jpg"
        path = os.path.join(image_dir, image_filename)
        try:
            img = Image.open(path).convert('RGB')
            label = row['label']
            base = row['image_id']

            for suffix, transformed in [('original', img), ('flipped', img.transpose(Image.FLIP_LEFT_RIGHT))]:
                filename = f"{base}_{suffix}.jpg"
                save_path = os.path.join(images_out, filename)
                resized = transformed.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                resized.save(save_path, 'JPEG', quality=95)

                metadata.append({"image_path": filename, "label": label, "split": split_value})
                feat = extract_features(resized)
                feat.update({"image_path": filename, "label": label, "split": split_value})
                features_list.append(feat)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    return features_list

# -----------------------------------------------------------------------------
# Report and Verify
# -----------------------------------------------------------------------------
def report_split(df, name):
    print(f"{name}: {len(df)} images, {df['lesion_id'].nunique()} lesions")
    print(df['label'].value_counts(), "\n")

def verify_no_leak(train_df, val_df, test_df):
    train_ids, val_ids, test_ids = set(train_df['lesion_id']), set(val_df['lesion_id']), set(test_df['lesion_id'])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        print("❌ Data leak detected between splits!")
    else:
        print("✅ No data leakage detected.")

# -----------------------------------------------------------------------------
# Dataset Generator
# -----------------------------------------------------------------------------
def create_dataset(df, output_dir, image_dir, avoid_leakage=False):
    print(f"\n🔧 Creating {'leak-free ' if avoid_leakage else 'leak-possible'} dataset at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if avoid_leakage:
        balanced = sample_balanced_no_leak(df)
        train, val, test = split_by_lesion_stratified(balanced)
    else:
        balanced = sample_balanced_leak(df)
        train, temp = train_test_split(balanced, test_size=0.3, random_state=42, stratify=balanced['label'])
        val, test = train_test_split(temp, test_size=2/3, random_state=42, stratify=temp['label'])

    if avoid_leakage:
        verify_no_leak(train, val, test)

    report_split(train, "TRAIN")
    report_split(val, "VAL")
    report_split(test, "TEST")

    metadata, features = [], []
    features += save_and_augment(train, output_dir, 0, metadata, image_dir)
    features += save_and_augment(val, output_dir, 1, metadata, image_dir)
    features += save_and_augment(test, output_dir, 2, metadata, image_dir)

    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "image_metadata.csv"), index=False)
    pd.DataFrame(features).to_csv(os.path.join(output_dir, "features_ludwig.csv"), index=False)

    images_dir = os.path.join(output_dir, "images")
    with zipfile.ZipFile(os.path.join(output_dir, "selected_images.zip"), 'w') as zf:
        for f in os.listdir(images_dir):
            zf.write(os.path.join(images_dir, f), arcname=f)
    shutil.rmtree(images_dir)

    print(f"✅ Done. {len(metadata)} image records written.")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="HAM10000 Preprocessing for Ludwig")
    parser.add_argument('--csv', type=str, required=True, help='Path to metadata CSV (e.g. HAM10000_metadata.csv)')
    parser.add_argument('--images', type=str, required=True, help='Path to folder with images')
    parser.add_argument('--dataset', choices=['leak', 'no_leak'], help='Choose dataset type')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    df = load_data(args.csv)

    if args.dataset == 'leak':
        create_dataset(df, OUTPUT_DIR_1, args.images, avoid_leakage=False)
    elif args.dataset == 'no_leak':
        create_dataset(df, OUTPUT_DIR_2, args.images, avoid_leakage=True)
    else:
        create_dataset(df, OUTPUT_DIR_1, args.images, avoid_leakage=False)
        create_dataset(df, OUTPUT_DIR_2, args.images, avoid_leakage=True)

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import zipfile
import argparse

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SAMPLES_PER_CLASS = 100
IMAGE_SIZE = (220, 220)
OUTPUT_DIR_1 = "./processed_data"          # leak-possible
OUTPUT_DIR_2 = "./processed_data_no_leak"  # leak-free

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Expect standard HAM10000-style columns: image_id, lesion_id, dx
    # Rename dx -> label
    if "dx" in df.columns:
        df = df.rename(columns={"dx": "label"})
    df = df.sort_values(by=["image_id", "label"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Leak-Possible Balanced Sampling (image-level)
# -----------------------------------------------------------------------------
def sample_balanced_leak(df):
    """Sample up to SAMPLES_PER_CLASS *images* per class (ignores lesion boundaries)."""
    sampled = []
    rng = np.random.default_rng(42)
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        n = min(SAMPLES_PER_CLASS, len(class_df))
        sampled.append(class_df.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True)


# -----------------------------------------------------------------------------
# Leak-Free Balanced Sampling (lesion-aware with fallback)
# -----------------------------------------------------------------------------
def sample_balanced_no_leak(df):
    """
    For each class:
      1) Take up to 1 image per *unique lesion_id* (random pick per lesion) until
         we reach SAMPLES_PER_CLASS or run out of lesions.
      2) If still short of SAMPLES_PER_CLASS, top up by sampling additional images
         from the remaining pool (can include multiple from same lesion).
    Returns: DataFrame with up to SAMPLES_PER_CLASS rows per class.
    """
    sampled = []
    rng = np.random.default_rng(42)

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]

        # Group by lesion
        grouped = class_df.groupby("lesion_id")

        # Shuffle lesion_ids deterministically
        lesion_ids = grouped.size().index.to_numpy()
        rng.shuffle(lesion_ids)

        picked_rows = []

        # Pass 1: one image per lesion
        for lid in lesion_ids:
            lesion_rows = class_df[class_df['lesion_id'] == lid]
            # Pick 1 image deterministically w/ seed
            one = lesion_rows.sample(n=1, random_state=42)
            picked_rows.append(one)
            if len(picked_rows) >= SAMPLES_PER_CLASS:
                break

        picked_df = pd.concat(picked_rows, ignore_index=True)

        # Pass 2: need to top up?
        if len(picked_df) < SAMPLES_PER_CLASS:
            need = SAMPLES_PER_CLASS - len(picked_df)
            # pool = all remaining images not already chosen
            pool = class_df[~class_df['image_id'].isin(picked_df['image_id'])]
            if len(pool) > 0:
                add_n = min(need, len(pool))
                picked_df = pd.concat(
                    [picked_df, pool.sample(n=add_n, random_state=42)],
                    ignore_index=True,
                )
            # else: class_df smaller than requested; accept shortfall

        sampled.append(picked_df)

    return pd.concat(sampled, ignore_index=True)


# -----------------------------------------------------------------------------
# Leak-Free Stratified Split (lesion-level, per class)
# -----------------------------------------------------------------------------
def split_by_lesion_stratified(df):
    """
    Split df into train/val/test by lesion_id *within each class*, preserving
    ~70/10/20 lesion distribution per class. All images from a lesion stay
    together in one split, so no leakage.
    """
    train_parts, val_parts, test_parts = [], [], []

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]

        # Unique lesion ids for this class
        lesion_ids = class_df['lesion_id'].unique()

        # If fewer than 10 lesions, fall back to proportionate but safe splitting
        if len(lesion_ids) < 3:
            # extreme small class; just put all in train to avoid empty splits
            train_parts.append(class_df)
            continue

        # 70/30 lesion split
        train_ids, temp_ids = train_test_split(
            lesion_ids, test_size=0.3, random_state=42
        )

        # 10/20 (val/test) from remaining lesions: 1/3 val, 2/3 test
        # Guard for very small leftover sets
        if len(temp_ids) == 1:
            val_ids = temp_ids
            test_ids = []
        else:
            val_ids, test_ids = train_test_split(
                temp_ids, test_size=2/3, random_state=42
            )

        train_parts.append(class_df[class_df['lesion_id'].isin(train_ids)])
        val_parts.append(class_df[class_df['lesion_id'].isin(val_ids)])
        test_parts.append(class_df[class_df['lesion_id'].isin(test_ids)])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else pd.DataFrame(columns=df.columns)
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
# Image Saving + Augmentation
# -----------------------------------------------------------------------------
def save_and_augment(df, output_dir, split_value, metadata, image_dir):
    """
    Save resized original + horizontal flip for each row.
    split_value: 0=train, 1=val, 2=test
    """
    images_out = os.path.join(output_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    for _, row in df.iterrows():
        image_filename = f"{row['image_id']}.jpg"
        path = os.path.join(image_dir, image_filename)
        try:
            img = Image.open(path).convert('RGB')
            resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            label = row['label']
            base = row['image_id']

            orig_filename = f"{base}_original.jpg"
            flip_filename = f"{base}_flipped.jpg"
            orig_path = os.path.join(images_out, orig_filename)
            flip_path = os.path.join(images_out, flip_filename)

            # Save original
            resized.save(orig_path, 'JPEG', quality=95)
            # Save flipped
            flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
            flipped.save(flip_path, 'JPEG', quality=95)

            metadata.append({"image_path": orig_filename, "label": label, "split": split_value})
            metadata.append({"image_path": flip_filename, "label": label, "split": split_value})
        except Exception as e:
            print(f"Skipping {path}: {e}")


# -----------------------------------------------------------------------------
# Metadata + Packaging
# -----------------------------------------------------------------------------
def save_metadata(metadata, output_dir):
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "image_metadata.csv"), index=False)


def zip_images(output_dir):
    images_dir = os.path.join(output_dir, "images")
    zip_path = os.path.join(output_dir, "selected_images.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in os.listdir(images_dir):
            zf.write(os.path.join(images_dir, file), arcname=file)


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------
def report_split(df, name):
    if df.empty:
        print(f"{name}: 0 images")
        return
    print(f"{name}: {len(df)} rows, {df['lesion_id'].nunique()} lesions")
    print(df['label'].value_counts())


def verify_no_leak(train_df, val_df, test_df):
    train_ids = set(train_df['lesion_id'])
    val_ids = set(val_df['lesion_id'])
    test_ids = set(test_df['lesion_id'])

    val_train_overlap = train_ids.intersection(val_ids)
    test_train_overlap = train_ids.intersection(test_ids)
    test_val_overlap = val_ids.intersection(test_ids)

    if val_train_overlap or test_train_overlap or test_val_overlap:
        print("❌ DATA LEAK DETECTED:")
        if val_train_overlap:
            print(f"  - {len(val_train_overlap)} lesion_id(s) shared between TRAIN and VAL")
        if test_train_overlap:
            print(f"  - {len(test_train_overlap)} lesion_id(s) shared between TRAIN and TEST")
        if test_val_overlap:
            print(f"  - {len(test_val_overlap)} lesion_id(s) shared between VAL and TEST")
    else:
        print("✅ Dataset is leak-free (no lesion_id overlap between splits)")


# -----------------------------------------------------------------------------
# Dataset Creation Wrapper
# -----------------------------------------------------------------------------
def create_dataset(
    df, output_dir, image_dir,
    avoid_leakage=False,
    samples_per_class=SAMPLES_PER_CLASS
):
    """
    Create dataset, save resized + flipped copies, metadata, and zip.
    If avoid_leakage=True: lesion-aware, class-balanced sampling & splitting.
    """
    print(f"\n🔧 Creating {'leak-free ' if avoid_leakage else ''}dataset at {output_dir}")

    # Make sure parent exists
    os.makedirs(output_dir, exist_ok=True)

    if avoid_leakage:
        balanced_df = sample_balanced_no_leak(df)
        train_df, val_df, test_df = split_by_lesion_stratified(balanced_df)
    else:
        balanced_df = sample_balanced_leak(df)
        train_df, temp_df = train_test_split(
            balanced_df,
            test_size=0.3,
            random_state=42,
            stratify=balanced_df['label'],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=2/3,
            random_state=42,
            stratify=temp_df['label'],
        )

    # Check leakage
    if avoid_leakage:
        verify_no_leak(train_df, val_df, test_df)

    # Report counts
    report_split(train_df, "TRAIN")
    report_split(val_df,   "VAL")
    report_split(test_df,  "TEST")

    # Save images + metadata
    metadata = []
    save_and_augment(train_df, output_dir, 0, metadata, image_dir)
    save_and_augment(val_df,   output_dir, 1, metadata, image_dir)
    save_and_augment(test_df,  output_dir, 2, metadata, image_dir)

    # Write metadata + zip
    save_metadata(metadata, output_dir)
    zip_images(output_dir)

    print(f"✅ Done: {len(metadata)} image entries saved (count includes flipped augmentations).")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess HAM10000 dataset with optional lesion-aware leak-free splitting."
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to HAM10000 metadata CSV file (must include image_id, lesion_id, dx or label).')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to folder containing HAM10000 images (image_id.jpg).')
    parser.add_argument('--dataset', type=str, choices=['leak', 'no_leak'],
                        help='Which dataset to generate. If omitted, both are generated.')
    parser.add_argument('--samples-per-class', type=int, default=SAMPLES_PER_CLASS,
                        help='Number of images per class to sample (before flip augmentation).')
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    global SAMPLES_PER_CLASS
    SAMPLES_PER_CLASS = args.samples_per_class  # allow override from CLI

    df = load_data(args.csv)

    if args.dataset == 'leak':
        create_dataset(df, OUTPUT_DIR_1, args.images, avoid_leakage=False,
                       samples_per_class=SAMPLES_PER_CLASS)
    elif args.dataset == 'no_leak':
        create_dataset(df, OUTPUT_DIR_2, args.images, avoid_leakage=True,
                       samples_per_class=SAMPLES_PER_CLASS)
    else:
        create_dataset(df, OUTPUT_DIR_1, args.images, avoid_leakage=False,
                       samples_per_class=SAMPLES_PER_CLASS)
        create_dataset(df, OUTPUT_DIR_2, args.images, avoid_leakage=True,
                       samples_per_class=SAMPLES_PER_CLASS)


if __name__ == "__main__":
    main()

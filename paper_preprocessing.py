import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import zipfile
import argparse

# Constants
SAMPLES_PER_CLASS = 100
IMAGE_SIZE = (220, 220)
OUTPUT_DIR_1 = "./processed_data"
OUTPUT_DIR_2 = "./processed_data_no_leak"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["image_id", "dx"])
    df = df.rename(columns={"dx": "label"})  # Use 'dx' as class label
    return df

def sample_balanced(df, stratify_by_lesion=False):
    sampled = []
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        if stratify_by_lesion:
            lesion_groups = class_df.groupby("lesion_id")
            unique_lesions = lesion_groups.first().reset_index()
            n = min(SAMPLES_PER_CLASS, len(unique_lesions))
            selected_lesions = unique_lesions.sample(n=n, random_state=42)
            sampled.append(selected_lesions)
        else:
            n = min(SAMPLES_PER_CLASS, len(class_df))
            sampled.append(class_df.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True)

def split_by_lesion(df):
    train_ids, temp_ids = train_test_split(df['lesion_id'].unique(), test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=2/3, random_state=42)

    train_df = df[df['lesion_id'].isin(train_ids)]
    val_df = df[df['lesion_id'].isin(val_ids)]
    test_df = df[df['lesion_id'].isin(test_ids)]

    return train_df, val_df, test_df

def save_and_augment(df, output_dir, split_value, metadata, image_dir):
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

            resized.save(orig_path, 'JPEG', quality=95)
            flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
            flipped.save(flip_path, 'JPEG', quality=95)

            metadata.append({"image_path": orig_filename, "label": label, "split": split_value})
            metadata.append({"image_path": flip_filename, "label": label, "split": split_value})
        except Exception as e:
            print(f"Skipping {path}: {e}")

def save_metadata(metadata, output_dir):
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "image_metadata.csv"), index=False)

def zip_images(output_dir):
    images_dir = os.path.join(output_dir, "images")
    zip_path = os.path.join(output_dir, "selected_images.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in os.listdir(images_dir):
            zf.write(os.path.join(images_dir, file), arcname=file)

def create_dataset(df, output_dir, image_dir, avoid_leakage=False):
    print(f"\n🔧 Creating {'leak-free ' if avoid_leakage else ''}dataset at {output_dir}")
    balanced_df = sample_balanced(df, stratify_by_lesion=avoid_leakage)

    if avoid_leakage:
        train_df, val_df, test_df = split_by_lesion(balanced_df)
    else:
        train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42, stratify=balanced_df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df['label'])

    # ✅ Check for lesion_id leakage
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

    metadata = []
    save_and_augment(train_df, output_dir, 0, metadata, image_dir)
    save_and_augment(val_df, output_dir, 1, metadata, image_dir)
    save_and_augment(test_df, output_dir, 2, metadata, image_dir)

    save_metadata(metadata, output_dir)
    zip_images(output_dir)

    print(f"✅ Done: {len(metadata)} images saved (with augmentation).")

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess HAM10000 dataset with and without lesion_id leakage.")
    parser.add_argument('--csv', type=str, required=True, help='Path to HAM10000 metadata CSV file')
    parser.add_argument('--images', type=str, required=True, help='Path to folder containing HAM10000 images')
    return parser.parse_args()

def main():
    args = parse_args()
    df = load_data(args.csv)

    # Dataset 1: Random split (may contain lesion_id leakage)
    create_dataset(df, OUTPUT_DIR_1, args.images, avoid_leakage=False)

    # Dataset 2: Lesion-aware split (no leakage)
    create_dataset(df, OUTPUT_DIR_2, args.images, avoid_leakage=True)

if __name__ == "__main__":
    main()

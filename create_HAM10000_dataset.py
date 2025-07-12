#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_image_splits(
    category_csv: str,
    output_train_csv: str,
    output_test_csv: str,
    images_per_class: int,
    random_state: int = 42
):
    """
    1) Load category CSV
    2) Sample `images_per_class` per label (no replacement expected here)
    3) Stratified 80/20 split by label
    4) Save train/test CSVs with headers image_path,label,split
    """
    df = pd.read_csv(category_csv)
    df_sampled = df.groupby('label').sample(
        n=images_per_class,
        replace=False,
        random_state=random_state
    )
    train_df, test_df = train_test_split(
        df_sampled,
        test_size=0.20,
        stratify=df_sampled['label'],
        random_state=random_state
    )
    train_df = train_df.copy(); train_df['split'] = 0
    test_df  = test_df.copy();  test_df['split']  = 1
    train_df[['image_path','label','split']].to_csv(output_train_csv, index=False)
    test_df[['image_path','label','split']].to_csv(output_test_csv, index=False)
    total = images_per_class * df_sampled['label'].nunique()
    print(f"\n=== Total images: {total} ({images_per_class} per class) ===")
    print(f"→ {len(train_df)} train images → {output_train_csv}")
    print(train_df['label'].value_counts().sort_index())
    print(f"→ {len(test_df)} test images → {output_test_csv}")
    print(test_df['label'].value_counts().sort_index())


def create_image_splits_no_leakage(
    category_csv: str,
    metadata_csv: str,
    output_train_csv: str,
    output_test_csv: str,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Leakage-free (no duplication) split by lesion
    1) Merge category with metadata for lesion_id
    2) Stratify-split lesions, assign splits to images
    3) Save CSVs
    """
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['lesion_id','image_id'])
    df = df.merge(meta, on='image_id', how='left')
    lesion_labels = df.groupby('lesion_id')['label'].agg(lambda x: x.mode()[0]).reset_index()
    train_lesions, test_lesions = train_test_split(
        lesion_labels,
        test_size=test_size,
        stratify=lesion_labels['label'],
        random_state=random_state
    )
    split_map = {lid:0 for lid in train_lesions['lesion_id']}
    split_map.update({lid:1 for lid in test_lesions['lesion_id']})
    df['split'] = df['lesion_id'].map(split_map)
    train_df = df[df['split']==0][['image_path','label','split']]
    test_df  = df[df['split']==1][['image_path','label','split']]
    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)
    print(f"\n=== Leakage-free (no duplication) split: {len(train_df)} train, {len(test_df)} test ===")
    print("Train label counts:\n", train_df['label'].value_counts().sort_index())
    print("Test  label counts:\n", test_df['label'].value_counts().sort_index())


def create_image_splits_no_leakage_with_replacement(
    category_csv: str,
    metadata_csv: str,
    output_train_csv: str,
    output_test_csv: str,
    images_per_class: int,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Leakage-free split by lesion, then upsample with replacement
    to reach desired images_per_class per class
    """
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['lesion_id','image_id'])
    df = df.merge(meta, on='image_id', how='left')
    lesion_labels = df.groupby('lesion_id')['label'].agg(lambda x: x.mode()[0]).reset_index()
    train_lesions, test_lesions = train_test_split(
        lesion_labels,
        test_size=test_size,
        stratify=lesion_labels['label'],
        random_state=random_state
    )
    split_map = {lid:0 for lid in train_lesions['lesion_id']}
    split_map.update({lid:1 for lid in test_lesions['lesion_id']})
    df['split'] = df['lesion_id'].map(split_map)
    train_all = df[df['split']==0]
    test_all  = df[df['split']==1]
    n_train = int(images_per_class * (1 - test_size))
    n_test  = images_per_class - n_train
    min_train = train_all['label'].value_counts().min()
    min_test  = test_all['label'].value_counts().min()
    if n_train > min_train:
        print(f"⚠️ Upsampling train to {n_train} per class (min available: {min_train})")
    if n_test > min_test:
        print(f"⚠️ Upsampling test to {n_test} per class (min available: {min_test})")
    train_df = train_all.groupby('label').apply(
        lambda g: g.sample(n=n_train, replace=(len(g)<n_train), random_state=random_state)
    ).reset_index(drop=True)
    test_df = test_all.groupby('label').apply(
        lambda g: g.sample(n=n_test, replace=(len(g)<n_test), random_state=random_state)
    ).reset_index(drop=True)
    train_df['split'] = 0
    test_df['split']  = 1
    train_df[['image_path','label','split']].to_csv(output_train_csv, index=False)
    test_df[['image_path','label','split']].to_csv(output_test_csv, index=False)
    print(f"\n=== Leakage-free with duplication: {len(train_df)} train, {len(test_df)} test ===")
    print(train_df['label'].value_counts().sort_index())
    print(test_df['label'].value_counts().sort_index())


def check_lesion_leakage(train_csv: str, test_csv: str, metadata_csv: str):
    """
    Check lesion-level leakage between train/test splits.
    Reads train and test CSVs, assigns split codes, and reports lesions in multiple splits.
    """
    df_train = pd.read_csv(train_csv)
    df_train['split'] = 0
    df_test = pd.read_csv(test_csv)
    df_test['split']  = 1
    df = pd.concat([df_train, df_test], ignore_index=True)
    df['image_id'] = df['image_path'].apply(
        lambda p: os.path.splitext(os.path.basename(p))[0]
    )
    meta = pd.read_csv(metadata_csv)[['image_id', 'lesion_id']]
    merged = df.merge(meta, on='image_id', how='left')
    counts = merged.groupby('lesion_id')['split'].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        print(f"⚠️ Leakage detected ({train_csv}, {test_csv}): {len(leaking)} lesion(s) span multiple splits.")
    else:
        print(f"✅ No leakage between {train_csv} and {test_csv}.")


if __name__ == '__main__':
    # Generate datasets
    create_image_splits(
        category_csv     = 'Ham10000_Category.csv',
        output_train_csv = 'Ham10000_train_700.csv',
        output_test_csv  = 'Ham10000_test_700.csv',
        images_per_class = 100,
        random_state     = 42
    )
    create_image_splits_no_leakage(
        category_csv     = 'Ham10000_Category.csv',
        metadata_csv     = 'HAM10000_metadata.csv',
        output_train_csv = 'Ham10000_train_noleak.csv',
        output_test_csv  = 'Ham10000_test_noleak.csv',
        test_size        = 0.20,
        random_state     = 42
    )
    create_image_splits_no_leakage_with_replacement(
        category_csv      = 'Ham10000_Category.csv',
        metadata_csv      = 'HAM10000_metadata.csv',
        output_train_csv  = 'Ham10000_train_noleak_dup.csv',
        output_test_csv   = 'Ham10000_test_noleak_dup.csv',
        images_per_class  = 200,
        test_size         = 0.20,
        random_state      = 42
    )

    # Check leakage for each dataset
    print()  # blank line
    check_lesion_leakage('Ham10000_train_700.csv',    'Ham10000_test_700.csv',    'HAM10000_metadata.csv')
    check_lesion_leakage('Ham10000_train_noleak.csv', 'Ham10000_test_noleak.csv', 'HAM10000_metadata.csv')
    check_lesion_leakage('Ham10000_train_noleak_dup.csv','Ham10000_test_noleak_dup.csv','HAM10000_metadata.csv')


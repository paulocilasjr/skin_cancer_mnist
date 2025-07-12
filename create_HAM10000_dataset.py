#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_combined_image_splits(
    category_csv: str,
    output_csv: str,
    images_per_class: int,
    random_state: int = 42
):
    """
    Create an 80/20 stratified image-level split,
    and save a single combined CSV with columns image_path,label,split.
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
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_csv(output_csv, index=False)
    print(f"Created {output_csv}: {len(train_df)} train + {len(test_df)} test = {len(combined)} rows")


def create_combined_image_splits_no_leakage(
    category_csv: str,
    metadata_csv: str,
    output_csv: str,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Create an 80/20 lesion-aware split (no leakage),
    and save a single combined CSV with columns image_path,label,split.
    """
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id', 'lesion_id'])
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
    combined = df[['image_path','label','split']]
    combined.to_csv(output_csv, index=False)
    counts = combined['split'].value_counts()
    print(f"Created {output_csv}: train={counts.get(0,0)}, test={counts.get(1,0)}")


def create_combined_image_splits_no_leakage_with_replacement(
    category_csv: str,
    metadata_csv: str,
    output_csv: str,
    images_per_class: int,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Create a lesion-aware split (no cross-lesion leakage),
    then upsample within train/test to reach images_per_class,
    and save a single combined CSV with columns image_path,label,split.
    """
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id', 'lesion_id'])
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
    train_df = train_all.groupby('label').apply(
        lambda g: g.sample(n=n_train, replace=(len(g)<n_train), random_state=random_state)
    ).reset_index(drop=True)
    test_df = test_all.groupby('label').apply(
        lambda g: g.sample(n=n_test, replace=(len(g)<n_test), random_state=random_state)
    ).reset_index(drop=True)
    train_df['split'] = 0
    test_df['split']  = 1
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_csv(output_csv, index=False)
    print(f"Created {output_csv}: train={len(train_df)}, test={len(test_df)} (total {len(combined)})")


def check_lesion_leakage(splits_csv: str, metadata_csv: str):
    """
    Check lesion-level leakage within a combined split file.
    Drops any pre-existing lesion_id to avoid merge collisions,
    then reports if a lesion appears in multiple splits.
    """
    df = pd.read_csv(splits_csv)
    # ensure only split-level info
    df = df.drop(columns=['lesion_id'], errors='ignore')
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id', 'lesion_id'])
    merged = df.merge(meta, on='image_id', how='left')
    counts = merged.groupby('lesion_id')['split'].nunique()
    leaking = counts[counts>1]
    if not leaking.empty:
        print(f"⚠️ Leakage detected in {splits_csv}: {len(leaking)} lesion(s) span multiple splits.")
    else:
        print(f"✅ No leakage detected in {splits_csv}.")

if __name__ == '__main__':
    # Generate and save only combined datasets
    create_combined_image_splits(
        category_csv='Ham10000_Category.csv',
        output_csv='Ham10000_700_combined.csv',
        images_per_class=100,
        random_state=42
    )
    create_combined_image_splits_no_leakage(
        category_csv='Ham10000_Category.csv',
        metadata_csv='HAM10000_metadata.csv',
        output_csv='Ham10000_noleak_combined.csv',
        test_size=0.20,
        random_state=42
    )
    create_combined_image_splits_no_leakage_with_replacement(
        category_csv='Ham10000_Category.csv',
        metadata_csv='HAM10000_metadata.csv',
        output_csv='Ham10000_noleak_dup_combined.csv',
        images_per_class=200,
        test_size=0.20,
        random_state=42
    )

    # Check leakage for each combined dataset
    print()
    check_lesion_leakage('Ham10000_700_combined.csv',    'HAM10000_metadata.csv')
    check_lesion_leakage('Ham10000_noleak_combined.csv', 'HAM10000_metadata.csv')
    check_lesion_leakage('Ham10000_noleak_dup_combined.csv','HAM10000_metadata.csv')

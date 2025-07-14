#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_combined_image_splits(
    category_csv: str,
    output_csv: str,
    images_per_class: int,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Create a stratified image‐level split into
      70% train (split=0),
      10% val   (split=1),
      20% test  (split=2),
    sampling exactly images_per_class total per label, then saving
    image_path,label,split to output_csv.
    """
    df = pd.read_csv(category_csv)
    # sample N images per class first
    df_sampled = df.groupby('label').sample(
        n=images_per_class,
        replace=False,
        random_state=random_state
    )

    # 1) carve off test
    train_val_df, test_df = train_test_split(
        df_sampled,
        test_size=test_size,
        stratify=df_sampled['label'],
        random_state=random_state
    )
    # 2) split train_val into train vs val
    val_frac = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac,
        stratify=train_val_df['label'],
        random_state=random_state
    )

    # assign split codes
    train_df = train_df.copy(); train_df['split'] = 0
    val_df   = val_df.copy();   val_df['split']   = 1
    test_df  = test_df.copy();  test_df['split']  = 2

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined.to_csv(output_csv, index=False)

    print(
        f"Created {output_csv}: "
        f"{len(train_df)} train + {len(val_df)} val + {len(test_df)} test = {len(combined)} rows"
    )


def create_combined_image_splits_no_leakage(
    category_csv: str,
    metadata_csv: str,
    output_csv: str,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Create a lesion‐aware split into 70/10/20 (0/1/2),
    ensuring no lesion_id appears in more than one split.
    """
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id', 'lesion_id'])
    df = df.merge(meta, on='image_id', how='left')

    # get one label per lesion by majority vote
    lesion_labels = (
        df.groupby('lesion_id')['label']
          .agg(lambda x: x.mode()[0])
          .reset_index()
    )

    # 1) carve off test‐lesions
    lesion_temp, test_lesions = train_test_split(
        lesion_labels,
        test_size=test_size,
        stratify=lesion_labels['label'],
        random_state=random_state
    )
    # 2) split remaining into train vs val
    val_frac = val_size / (1 - test_size)
    train_lesions, val_lesions = train_test_split(
        lesion_temp,
        test_size=val_frac,
        stratify=lesion_temp['label'],
        random_state=random_state
    )

    # build map lesion_id -> split code
    split_map = {lid: 0 for lid in train_lesions['lesion_id']}
    split_map.update({lid: 1 for lid in val_lesions['lesion_id']})
    split_map.update({lid: 2 for lid in test_lesions['lesion_id']})

    df['split'] = df['lesion_id'].map(split_map)

    combined = df[['image_path','label','split']]
    combined.to_csv(output_csv, index=False)

    counts = combined['split'].value_counts().sort_index()
    print(
        f"Created {output_csv}: "
        f"train={counts.get(0,0)}, val={counts.get(1,0)}, test={counts.get(2,0)}"
    )


def create_combined_image_splits_no_leakage_with_replacement(
    category_csv: str,
    metadata_csv: str,
    output_csv: str,
    images_per_class: int,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Lesion‐aware 70/10/20 split (0/1/2), then upsample each split
    so that, for each label, the total count across train+val+test
    equals images_per_class.
    """
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id', 'lesion_id'])
    df = df.merge(meta, on='image_id', how='left')

    lesion_labels = (
        df.groupby('lesion_id')['label']
          .agg(lambda x: x.mode()[0])
          .reset_index()
    )

    # split lesions same as above
    lesion_temp, test_lesions = train_test_split(
        lesion_labels,
        test_size=test_size,
        stratify=lesion_labels['label'],
        random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    train_lesions, val_lesions = train_test_split(
        lesion_temp,
        test_size=val_frac,
        stratify=lesion_temp['label'],
        random_state=random_state
    )

    split_map = {lid: 0 for lid in train_lesions['lesion_id']}
    split_map.update({lid: 1 for lid in val_lesions['lesion_id']})
    split_map.update({lid: 2 for lid in test_lesions['lesion_id']})
    df['split'] = df['lesion_id'].map(split_map)

    train_all = df[df['split']==0]
    val_all   = df[df['split']==1]
    test_all  = df[df['split']==2]

    # compute per‐split targets
    n_train = int(images_per_class * (1 - val_size - test_size))
    n_val   = int(images_per_class * val_size)
    n_test  = images_per_class - n_train - n_val

    # sample/upsample within each
    train_df = (
        train_all
        .groupby('label')
        .apply(lambda g: g.sample(n=n_train, replace=(len(g)<n_train), random_state=random_state))
        .reset_index(drop=True)
    )
    val_df = (
        val_all
        .groupby('label')
        .apply(lambda g: g.sample(n=n_val, replace=(len(g)<n_val), random_state=random_state))
        .reset_index(drop=True)
    )
    test_df = (
        test_all
        .groupby('label')
        .apply(lambda g: g.sample(n=n_test, replace=(len(g)<n_test), random_state=random_state))
        .reset_index(drop=True)
    )

    train_df['split'] = 0
    val_df['split']   = 1
    test_df['split']  = 2

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined.to_csv(output_csv, index=False)

    print(
        f"Created {output_csv}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
        f"(total {len(combined)})"
    )


def check_lesion_leakage(splits_csv: str, metadata_csv: str):
    """
    Ensure no lesion spans >1 split (0/1/2).
    """
    df = pd.read_csv(splits_csv).drop(columns=['lesion_id'], errors='ignore')
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id','lesion_id'])
    merged = df.merge(meta, on='image_id', how='left')

    counts = merged.groupby('lesion_id')['split'].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        print(f"⚠️ Leakage detected in {splits_csv}: {len(leaking)} lesion(s) span multiple splits.")
    else:
        print(f"✅ No leakage detected in {splits_csv}.")


if __name__ == '__main__':
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
        random_state=42
    )
    create_combined_image_splits_no_leakage_with_replacement(
        category_csv='Ham10000_Category.csv',
        metadata_csv='HAM10000_metadata.csv',
        output_csv='Ham10000_noleak_dup_combined.csv',
        images_per_class=200,
        random_state=42
    )

    print()
    check_lesion_leakage('Ham10000_700_combined.csv',      'HAM10000_metadata.csv')
    check_lesion_leakage('Ham10000_noleak_combined.csv',   'HAM10000_metadata.csv')
    check_lesion_leakage('Ham10000_noleak_dup_combined.csv','HAM10000_metadata.csv')

#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def balanced_sample(df, n, random_state):
    """
    Sample exactly n images per label, with replacement if needed.
    """
    return (
        df
        .groupby('label')
        .apply(lambda g: g.sample(n=n, replace=(len(g) < n), random_state=random_state))
        .reset_index(drop=True)
    )

def paper_split_leak(
    category_csv: str,
    output_csv: str,
    images_per_class: int = 100,
    random_state: int = 42
):
    """
    PAPER-LEAK:
    Sample 100 images/class (700 total), then do an 80/20 random stratified split.
    Lesions will leak.
    """
    df = pd.read_csv(category_csv)
    df_sampled = df.groupby('label').sample(
        n=images_per_class, replace=False, random_state=random_state
    )

    train, test = train_test_split(
        df_sampled,
        test_size=0.20,
        stratify=df_sampled['label'],
        random_state=random_state
    )

    train = train.copy(); train['split'] = 0
    test  = test.copy();  test['split']  = 1

    out = pd.concat([train, test], ignore_index=True)
    out = out[['image_path','label','split']]             # keep only needed columns
    out.to_csv(output_csv, index=False)

    counts = out['split'].value_counts().sort_index()
    print(f"[PAPER-LEAK]    {output_csv}: train={counts.get(0,0)}, test={counts.get(1,0)}")
    return output_csv

def paper_split_noleak(
    category_csv: str,
    metadata_csv: str,
    output_csv: str,
    images_per_class: int = 100,
    random_state: int = 42
):
    """
    PAPER-NOLEAK:
    Sample 100 images/class, then do an 80/20 lesion-aware split,
    and within each split sample exactly 80 images/class for train
    and 20 images/class for test.
    """
    # 1) read & merge metadata
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id','lesion_id'])
    df = df.merge(meta, on='image_id', how='left')

    # 2) sample 100 images per class
    df_sampled = df.groupby('label').apply(
        lambda g: g.sample(n=images_per_class, replace=False, random_state=random_state)
    ).reset_index(drop=True)

    # 3) build lesion → label map for sampled set
    lesion_labels = (
        df_sampled.groupby('lesion_id')['label']
                 .agg(lambda x: x.mode()[0])
                 .reset_index()
    )

    # 4) lesion-aware 80/20 split on lesions
    lesion_temp, test_lesions = train_test_split(
        lesion_labels,
        test_size=0.20,
        stratify=lesion_labels['label'],
        random_state=random_state
    )
    train_lesions = lesion_temp

    # 5) assign splits to the sampled DataFrame
    train_ids = set(train_lesions['lesion_id'])
    df_sampled['split'] = df_sampled['lesion_id'].map(
        lambda lid: 0 if lid in train_ids else 1
    )

    # 6) fixed-count sampling within each split
    n_train = int(images_per_class * 0.80)  # 80 per class
    n_test  = images_per_class - n_train   # 20 per class

    train_df = balanced_sample(df_sampled[df_sampled['split']==0], n_train, random_state)
    test_df  = balanced_sample(df_sampled[df_sampled['split']==1], n_test, random_state)

    train_df = train_df.copy(); train_df['split'] = 0
    test_df  = test_df.copy();  test_df['split']  = 1

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined[['image_path','label','split']]    # keep only needed columns
    combined.to_csv(output_csv, index=False)

    counts = combined['split'].value_counts().sort_index()
    print(f"[PAPER-NOLEAK] {output_csv}: train={counts.get(0,0)}, test={counts.get(1,0)}")
    return output_csv

def full_split_leak(
    category_csv: str,
    output_csv: str,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    FULL-LEAK:
    Use all images, do a 70/10/20 stratified split by label.
    Lesions will leak.
    """
    df = pd.read_csv(category_csv)
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_frac,
        stratify=train_val['label'],
        random_state=random_state
    )

    train = train.copy(); train['split'] = 0
    val   = val.copy();   val['split']   = 1
    test  = test.copy();  test['split']  = 2

    combined = pd.concat([train, val, test], ignore_index=True)
    combined = combined[['image_path','label','split']]    # keep only needed columns
    combined.to_csv(output_csv, index=False)

    counts = combined['split'].value_counts().sort_index()
    print(f"[FULL-LEAK]     {output_csv}: train={counts.get(0,0)}, val={counts.get(1,0)}, test={counts.get(2,0)}")
    return output_csv

def full_split_noleak(
    category_csv: str,
    metadata_csv: str,
    output_csv: str,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    FULL-NOLEAK:
    Use all images, lesion-aware 70/10/20 split (no lesion crosses splits).
    """
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id','lesion_id'])
    df = df.merge(meta, on='image_id', how='left')

    lesion_labels = (
        df.groupby('lesion_id')['label']
          .agg(lambda x: x.mode()[0])
          .reset_index()
    )

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

    train_ids = set(train_lesions['lesion_id'])
    val_ids   = set(val_lesions  ['lesion_id'])
    df['split'] = df['lesion_id'].map(
        lambda lid: 0 if lid in train_ids else (1 if lid in val_ids else 2)
    )

    out = df[['image_path','label','split']]               # keep only needed columns
    out.to_csv(output_csv, index=False)

    counts = out['split'].value_counts().sort_index()
    print(f"[FULL-NOLEAK]  {output_csv}: train={counts.get(0,0)}, val={counts.get(1,0)}, test={counts.get(2,0)}")
    return output_csv

def check_leakage(splits_csv: str, metadata_csv: str):
    """
    Print whether any lesion_id spans more than one split.
    """
    df = pd.read_csv(splits_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id','lesion_id'])
    merged = df.merge(meta, on='image_id', how='left')

    if 'lesion_id' not in merged.columns:
        print(f"{os.path.basename(splits_csv)}: cannot check leakage (no lesion_id)")
        return

    counts = merged.groupby('lesion_id')['split'].nunique()
    leaking = counts[counts > 1]
    if leaking.empty:
        print(f"{os.path.basename(splits_csv)}: ✅ no leakage")
    else:
        print(f"{os.path.basename(splits_csv)}: ⚠️ LEAKAGE ({len(leaking)} lesion(s))")

if __name__ == "__main__":
    CAT_CSV  = "Ham10000_Category.csv"
    META_CSV = "HAM10000_metadata.csv"

    p_leak   = paper_split_leak(CAT_CSV,  "HAM10000_paper_leak.csv")
    p_noleak = paper_split_noleak(CAT_CSV, META_CSV, "HAM10000_paper_noleak.csv")

    f_leak   = full_split_leak(CAT_CSV,  "HAM10000_full_leak.csv")
    f_noleak = full_split_noleak(CAT_CSV, META_CSV, "HAM10000_full_noleak.csv")

    print("\nLeakage checks:")
    for fname in [p_leak, p_noleak, f_leak, f_noleak]:
        check_leakage(fname, META_CSV)

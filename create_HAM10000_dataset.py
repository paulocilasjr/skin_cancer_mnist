#!/usr/bin/env python3
import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

# suppress that scipy._lib.messagestream warning
warnings.filterwarnings(
    "ignore",
    message="scipy._lib.messagestream.MessageStream size changed"
)

def balanced_sample(df, n, random_state):
    """
    Sample exactly n images per label, with replacement if needed.
    """
    return (
        df
        .groupby('label')
        .apply(lambda g: g.sample(
            n=n,
            replace=(len(g) < n),
            random_state=random_state
        ))
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
    • Sample N images/class (100 → 700 total, or 200 → 1400 total),
      with replacement if N > available.
    • 80% train (split=0), 20% test (split=2).
    """
    df = pd.read_csv(category_csv)

    # now sample with replace whenever group is too small
    df_sampled = (
        df
        .groupby('label')
        .apply(lambda g: g.sample(
            n=images_per_class,
            replace=(len(g) < images_per_class),
            random_state=random_state
        ))
        .reset_index(drop=True)
    )

    train, test = train_test_split(
        df_sampled,
        test_size=0.20,
        stratify=df_sampled['label'],
        random_state=random_state
    )

    train = train.copy(); train['split'] = 0
    test  = test.copy();  test ['split'] = 2

    out = pd.concat([train, test], ignore_index=True)[['image_path','label','split']]
    out.to_csv(output_csv, index=False)

    counts = out['split'].value_counts().sort_index()
    print(f"[PAPER-LEAK]    {output_csv}: train={counts.get(0,0)}, test={counts.get(2,0)}")
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
    • Sample N images/class (100 → 700 total, or 200 → 1400 total),
      with replacement if N > available.
    • Lesion-aware 80/20 split; then fixed-count sampling
      with replacement as needed.
    """
    # 1) merge metadata
    df = pd.read_csv(category_csv)
    df['image_id'] = df['image_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    meta = pd.read_csv(metadata_csv, usecols=['image_id','lesion_id'])
    df = df.merge(meta, on='image_id', how='left')

    # 2) sample per class (with replacement if group too small)
    df_sampled = (
        df
        .groupby('label')
        .apply(lambda g: g.sample(
            n=images_per_class,
            replace=(len(g) < images_per_class),
            random_state=random_state
        ))
        .reset_index(drop=True)
    )

    # 3) lesion→majority-label
    lesion_labels = (
        df_sampled.groupby('lesion_id')['label']
                 .agg(lambda x: x.mode()[0])
                 .reset_index()
    )

    # 4) 80/20 lesion-aware split
    lesion_temp, test_lesions = train_test_split(
        lesion_labels,
        test_size=0.20,
        stratify=lesion_labels['label'],
        random_state=random_state
    )
    train_lesions = lesion_temp

    # 5) provisional split assignment
    train_ids = set(train_lesions['lesion_id'])
    df_sampled['split'] = df_sampled['lesion_id'].map(
        lambda lid: 0 if lid in train_ids else 2
    )

    # 6) fixed-count balanced sampling within each split
    n_train = int(images_per_class * 0.80)
    n_test  = images_per_class - n_train

    train_df = balanced_sample(df_sampled[df_sampled['split']==0], n_train, random_state).assign(split=0)
    test_df  = balanced_sample(df_sampled[df_sampled['split']==2], n_test,  random_state).assign(split=2)

    out = pd.concat([train_df, test_df], ignore_index=True)[['image_path','label','split']]
    out.to_csv(output_csv, index=False)

    counts = out['split'].value_counts().sort_index()
    print(f"[PAPER-NOLEAK] {output_csv}: train={counts.get(0,0)}, test={counts.get(2,0)}")
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
    • All images, stratified 70/10/20 by label:
      – train (split=0), val (split=1), test (split=2)
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

    out = pd.concat([train, val, test], ignore_index=True)[['image_path','label','split']]
    out.to_csv(output_csv, index=False)

    counts = out['split'].value_counts().sort_index()
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
    • All images, lesion-aware 70/10/20 (no lesion crosses splits).
    • Splits: train=0, val=1, test=2.
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

    out = df[['image_path','label','split']]
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

    # PAPER-LEAK (orig vs aug)
    paper_split_leak(CAT_CSV,  "HAM10000_paper_leak.csv",     images_per_class=100)
    paper_split_leak(CAT_CSV,  "HAM10000_paper_leak_aug.csv", images_per_class=200)

    # PAPER-NOLEAK (orig vs aug)
    paper_split_noleak(CAT_CSV, META_CSV, "HAM10000_paper_noleak.csv",     images_per_class=100)
    paper_split_noleak(CAT_CSV, META_CSV, "HAM10000_paper_noleak_aug.csv", images_per_class=200)

    # FULL splits
    full_split_leak(CAT_CSV,       "HAM10000_full_leak.csv")
    full_split_noleak(CAT_CSV, META_CSV, "HAM10000_full_noleak.csv")

    print("\nLeakage checks:")
    for fname in [
        "HAM10000_paper_leak.csv",
        "HAM10000_paper_leak_aug.csv",
        "HAM10000_paper_noleak.csv",
        "HAM10000_paper_noleak_aug.csv",
        "HAM10000_full_leak.csv",
        "HAM10000_full_noleak.csv"
    ]:
        check_leakage(fname, META_CSV)

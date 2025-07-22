import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import shutil

def create_paper_methodology_split():
    """
    Implements the exact methodology from the paper:
    1. Select 100 images per class (to handle imbalance)
    2. Apply horizontal flip augmentation (100->200 per class)
    3. Split into train/val/test (70/10/20)
    4. Create new CSV and copy/augment images
    """
    
    original_csv = "/Volumes/SP PHD U3/New_Ham10000/Data/Ham10000_Category.csv"
    original_images_dir = "/Volumes/SP PHD U3/New_Ham10000/extracted_images/HAM10000_all_images"
    
    df = pd.read_csv(original_csv)
    print(f"Original dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    # select 100 images per class (paper's approach to handle imbalance)
    selected_data = []
    
    for label in sorted(df['label'].unique()):
        class_data = df[df['label'] == label]
        print(f"\nClass {label}: {len(class_data)} images available")
        
        # select 100 images per class 
        n_select = min(100, len(class_data))
        selected = class_data.sample(n=n_select, random_state=42)
        selected_data.append(selected)
        print(f"Selected {len(selected)} images for class {label}")
    
    # combine selected data
    selected_df = pd.concat(selected_data, ignore_index=True)
    print(f"\nTotal selected images: {len(selected_df)}")
    print(f"Selected distribution:\n{selected_df['label'].value_counts().sort_index()}")
    
    # create train/val/test splits (70/10/20) before augmentation
    # this ensures original images are properly distributed
    train_data = []
    val_data = []
    test_data = []
    
    for label in sorted(selected_df['label'].unique()):
        class_data = selected_df[selected_df['label'] == label]
        
        # first split: 70% train, 30% temp
        train_subset, temp_subset = train_test_split(
            class_data, test_size=0.3, random_state=42, stratify=None
        )
        
        # second split: from 30% -> 10% val, 20% test (1/3 and 2/3 of 30%)
        val_subset, test_subset = train_test_split(
            temp_subset, test_size=0.667, random_state=42, stratify=None
        )
        
        train_data.append(train_subset)
        val_data.append(val_subset)
        test_data.append(test_subset)
        
        print(f"Class {label}: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")
    
    # combine splits
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"\nSplit sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # create output directories
    output_dir = "/Volumes/SP PHD U3/New_Ham10000/paper_processed"
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    final_data = []
    
    def process_and_augment(df_subset, split_name):
        """process images and apply horizontal flip augmentation"""
        processed_data = []
        
        for idx, row in df_subset.iterrows():
            original_path = row['image_path']
            full_original_path = os.path.join("/Volumes/SP PHD U3/New_Ham10000/extracted_images", original_path)
            
            if not os.path.exists(full_original_path):
                print(f"Warning: Image not found: {full_original_path}")
                continue
            
            try:
                image = Image.open(full_original_path)
                image = image.convert('RGB')
            except Exception as e:
                print(f"Warning: Could not read image: {full_original_path}, Error: {e}")
                continue
            
            # get original filename without extension
            original_filename = os.path.basename(original_path)
            name_without_ext = os.path.splitext(original_filename)[0]
            
            # save original image (resized to 220x220 for ML as per paper)
            original_resized = image.resize((220, 220), Image.Resampling.LANCZOS)
            original_output_path = os.path.join(output_images_dir, f"{name_without_ext}_original.jpg")
            original_resized.save(original_output_path, 'JPEG', quality=95)
            
            # map split names to numeric values like original dataset
            split_mapping = {'train': 0, 'validation': 1, 'test': 2}
            split_numeric = split_mapping[split_name]
            
            # add original to data
            processed_data.append({
                'image_path': f"images/{name_without_ext}_original.jpg",
                'label': row['label'],
                'split': split_numeric
            })
            
            # create and save horizontally flipped image (paper's augmentation method)
            flipped_image = original_resized.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            flipped_output_path = os.path.join(output_images_dir, f"{name_without_ext}_flipped.jpg")
            flipped_image.save(flipped_output_path, 'JPEG', quality=95)
            
            # add flipped to data
            processed_data.append({
                'image_path': f"images/{name_without_ext}_flipped.jpg",
                'label': row['label'],
                'split': split_numeric
            })
        
        return processed_data
    
    # process each split
    print("\nProcessing and augmenting images...")
    print("Processing training set...")
    train_processed = process_and_augment(train_df, 'train')
    
    print("Processing validation set...")
    val_processed = process_and_augment(val_df, 'validation')
    
    print("Processing test set...")
    test_processed = process_and_augment(test_df, 'test')
    
    # combine all processed data
    final_data = train_processed + val_processed + test_processed
    
    final_df = pd.DataFrame(final_data)
    
    output_csv = os.path.join(output_dir, "paper_methodology_dataset.csv")
    final_df.to_csv(output_csv, index=False)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total images created: {len(final_df)}")
    print(f"Output directory: {output_dir}")
    print(f"CSV file: {output_csv}")
    
    print(f"\nSplit distribution:")
    split_counts = final_df['split'].value_counts()
    print(split_counts)
    
    print(f"\nLabel distribution per split:")
    split_names = {0: 'TRAIN', 1: 'VALIDATION', 2: 'TEST'}
    for split_num in [0, 1, 2]:
        split_data = final_df[final_df['split'] == split_num]
        print(f"\n{split_names[split_num]} (split={split_num}):")
        print(split_data['label'].value_counts().sort_index())
    
    print(f"\nAugmentation summary:")
    print(f"Total images: {len(final_df)}")
    print("Each class has been doubled through horizontal flip augmentation")
        
    return final_df

if __name__ == "__main__":
    result_df = create_paper_methodology_split()
    print("\nPaper methodology preprocessing completed successfully!") 

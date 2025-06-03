import pandas as pd
import hashlib
import os

def assign_split(image_id, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Assigns an image to a split (training, validation, testing) based on its hash.
    Uses the original Ham10000 image_id for hashing.
    Returns 0 for training, 1 for validation, 2 for testing.
    """
    if not (0.999 < train_ratio + val_ratio + test_ratio < 1.001):
        print(f"Error: Train ({train_ratio}), validation ({val_ratio}), and test ({test_ratio}) ratios must sum to 1.")
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    
    hasher = hashlib.sha256(str(image_id).encode('utf-8'))
    hash_val = int(hasher.hexdigest(), 16)
    remainder = hash_val % 100
    
    if remainder < train_ratio * 100:
        return 0  # train
    elif remainder < (train_ratio + val_ratio) * 100:
        return 1  # validation
    else:
        return 2  # testing

def create_category_split_file(metadata_path, output_path):
    """
    Creates the Ham10000_Category.csv file with image_id, Ludwig-compatible label, 
    and numerical split (0:train, 1:val, 2:test).
    """
    print(f"--- Starting to create category split file ---")
    print(f"Reading metadata from: {metadata_path}")
    
    dx_to_label = {
        'akiec': 0,
        'bcc': 1,
        'bkl': 2,
        'df': 3,
        'nv': 4,
        'vasc': 5,
        'mel': 6
    }
    
    try:
        metadata_df = pd.read_csv(metadata_path)
        print(f"Successfully read {len(metadata_df)} rows from metadata.")
        
        output_data = []
        print(f"Processing {len(metadata_df)} rows to assign labels and splits...")
        
        for index, row in metadata_df.iterrows():
            if (index + 1) % 1000 == 0:
                print(f"Processing row {index + 1}...")
                
            original_image_id = str(row['image_id'])
            dx_type = row['dx']
            
            label = dx_to_label.get(dx_type, -1)
            if label == -1:
                print(f"Warning: dx_type '{dx_type}' for image '{original_image_id}' not found in label map. Assigning -1.")

            split = assign_split(original_image_id)
            
            output_image_id_format = f"HAM10000_all_images/{original_image_id}.jpg"
            
            output_data.append({
                'image_id': output_image_id_format,
                'label': label,
                'split': split
            })
        
        print("Finished processing rows.")
        output_df = pd.DataFrame(output_data)
        
        final_columns_order = ['image_id', 'label', 'split']
        output_df = output_df[final_columns_order]
        
        print(f"Attempting to write DataFrame to: {output_path}")
        output_df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path} with {len(output_df)} entries.")
        
        print("\nSplit distribution:")
        split_counts = output_df['split'].value_counts().sort_index()
        split_names = {0: 'Training (0)', 1: 'Validation (1)', 2: 'Testing (2)'}
        total_images = len(output_df)
        if total_images > 0:
            for split_val, count in split_counts.items():
                percentage = (count / total_images) * 100
                print(f"  {split_names.get(split_val, f'Unknown Split ({split_val})')}: {count} images ({percentage:.2f}%)")
        else:
            print("  No images to calculate split distribution.")

        print("\nLabel distribution:")
        label_counts = output_df['label'].value_counts().sort_index()
        label_to_dx_str = {v: k for k, v in dx_to_label.items()} 
        if total_images > 0:
            for label_val, count in label_counts.items():
                dx_name = label_to_dx_str.get(label_val, f'Unknown Label ({label_val})')
                percentage = (count / total_images) * 100
                print(f"  Label {label_val} ({dx_name}): {count} images ({percentage:.2f}%)")
        else:
            print("  No images to calculate label distribution.")
            
        return True
        
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("--- Starting HAM10000 Category and Split Generation Script ---")
    
    base_dir = "/Volumes/SP PHD U3/New_Ham10000"
    metadata_file = os.path.join(base_dir, "HAM10000_metadata.csv")
    output_file = os.path.join(base_dir, "Ham10000_Category.csv")

    print(f"Using base directory: {base_dir}")
    print(f"Input metadata file: {metadata_file}")
    print(f"Output CSV file: {output_file}")

    success = create_category_split_file(metadata_file, output_file)
    
    if success:
        print("\n--- Script completed successfully. ---")
    else:
        print("\n--- Script finished with errors. Please check logs. ---")

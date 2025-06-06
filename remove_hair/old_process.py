import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import zipfile

def remove_hair(image):
    """
    Remove hair from skin lesion images using blackhat transform and inpainting
    as described in the research paper.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    inpainted = cv2.inpaint(image, thresh, 5, cv2.INPAINT_TELEA)
    
    return inpainted

def preprocess_dataset(csv_path, image_dir, output_dir):
    """
    Preprocess all images in the dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(csv_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):

        image_filename = os.path.basename(row['image_id'])
        image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        processed_image = remove_hair(image)
        
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        
        output_filename = image_filename
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, processed_image)
    
    print(f"Preprocessing completed. Processed images saved to {output_dir}")

def visualize_examples(csv_path, image_dir, processed_dir, num_examples=5):
    """
    Visualize a few examples of original and processed images.
    """
    df = pd.read_csv(csv_path)
    
    samples = df.sample(num_examples)
    
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3*num_examples))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        # Extract just the filename from the full path
        image_filename = os.path.basename(row['image_id'])
        image_path = os.path.join(image_dir, image_filename)
        processed_path = os.path.join(processed_dir, image_filename)
        
        # Read images
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        proc_img = cv2.imread(processed_path)
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        
        # Display images
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original (Class: {row['label']}, Split: {row['split']})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(proc_img)
        axes[i, 1].set_title("Hair Removed")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(processed_dir, 'preprocessing_examples.png'))
    plt.show()

def clean_macos_files(directory):
    """
    Remove macOS system files from a directory and its subdirectories.
    """
    print("Cleaning macOS system files...")
    
    # List of patterns to remove
    macos_patterns = ['.DS_Store', '._*', '.AppleDouble', '__MACOSX']
    count = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # First, remove any matching directories
        for pattern in macos_patterns:
            for d in dirs[:]:  # Use a copy since we're modifying dirs
                if d == pattern or d.startswith('._'):
                    dir_path = os.path.join(root, d)
                    print(f"Removing directory: {dir_path}")
                    shutil.rmtree(dir_path, ignore_errors=True)
                    dirs.remove(d)
                    count += 1
        
        # Then remove any matching files
        for pattern in macos_patterns:
            for f in files:
                if f == pattern or f.startswith('._'):
                    file_path = os.path.join(root, f)
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)
                    count += 1
    
    # Special case for top-level .DS_Store which might be missed
    ds_store_path = os.path.join(directory, '.DS_Store')
    if os.path.exists(ds_store_path):
        os.remove(ds_store_path)
        count += 1
    
    macos_dir = os.path.join(directory, '__MACOSX')
    if os.path.exists(macos_dir):
        shutil.rmtree(macos_dir, ignore_errors=True)
        count += 1
    
    print(f"Cleaned {count} macOS system files/directories")

def create_zip_file(dir_path, zip_path):
    """
    Create a zip file from the contents of a directory, 
    excluding macOS system files and other hidden files.
    """
    print(f"Creating zip file at {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files in the directory
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # skip macOS hidden files
                if file.startswith('.') or file == 'DS_Store' or file == '__MACOSX':
                    continue
                    
                # create complete filepath of file in directory
                file_path = os.path.join(root, file)
                
                # skip any other macOS system files
                if '/__MACOSX/' in file_path or '/.DS_Store' in file_path:
                    continue
                
                zipf.write(
                    file_path, 
                    os.path.relpath(file_path, os.path.join(dir_path, '..'))
                )
    
    print(f"Zip file created successfully at {zip_path}")
    # get size of zip file in MB
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Zip file size: {zip_size_mb:.2f} MB")

def create_processed_dataset_with_csv(csv_path, processed_dir, output_zip_path, fixed_csv_path):
    """
    Create a new CSV file with paths to processed images and zip everything.
    Maintains the same split column and distribution from the original CSV.
    """
    df = pd.read_csv(csv_path)
    # update image paths to point to processed directory
    df['image_id'] = df['image_id'].apply(lambda x: os.path.basename(x))
    # Save the processed CSV in the processed_dir
    processed_csv_path = os.path.join(processed_dir, 'old_processed.csv')
    df.to_csv(processed_csv_path, index=False)
    print(f"\nProcessed CSV saved to {processed_csv_path}")
    # Clean macOS files before zipping
    clean_macos_files(processed_dir)
    # Create the zip file
    create_zip_file(processed_dir, output_zip_path)
    # Create the fixed CSV with correct image paths for use with the zip
    df['image_id'] = df['image_id'].apply(lambda x: f'old_processed/{x}')
    df.to_csv(fixed_csv_path, index=False)
    print(f"Saved fixed CSV to {fixed_csv_path}")

if __name__ == "__main__":
    # paths
    csv_path = '/Volumes/SP PHD U3/New_Ham10000/HAM10000_Category.csv'
    image_dir = '/Volumes/SP PHD U3/New_Ham10000/extracted_images/HAM10000_all_images'
    output_dir = '/Volumes/SP PHD U3/New_Ham10000/remove_hair/old_processed'
    output_zip_path = '/Volumes/SP PHD U3/New_Ham10000/remove_hair/old_processed.zip'
    fixed_csv_path = '/Volumes/SP PHD U3/New_Ham10000/remove_hair/old_processed_fixed.csv'

    # process the dataset
    preprocess_dataset(csv_path, image_dir, output_dir)
    # visualize examples
    visualize_examples(csv_path, image_dir, output_dir)
    # create processed CSV, zip file, and fixed CSV
    create_processed_dataset_with_csv(csv_path, output_dir, output_zip_path, fixed_csv_path) 
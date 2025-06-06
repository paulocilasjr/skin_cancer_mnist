import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time
import multiprocessing

# Paths
csv_path = '/Volumes/SP PHD U3/New_Ham10000/Ham10000_Category.csv'
image_dir = '/Volumes/SP PHD U3/New_Ham10000/extracted_images/HAM10000_all_images'
processed_dir = '/Volumes/SP PHD U3/New_Ham10000/old_processed'
output_png = '/Volumes/SP PHD U3/New_Ham10000/best_hair_removal_examples.png'

def compute_hair_score(image_path):
    if not os.path.exists(image_path):
        return (0, image_path)
    img = cv2.imread(image_path)
    if img is None:
        return (0, image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    hair_score = np.sum(blackhat > 10)
    return (hair_score, image_path)

if __name__ == "__main__":
    total_start = time.time()
    df = pd.read_csv(csv_path)
    # only use the first 500 images 
    image_paths = [os.path.join(image_dir, os.path.basename(row['image_id'])) for _, row in df.head(500).iterrows()]

    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPU cores for parallel processing.")
    start_time = time.time()
    # use all CPU cores for hair score computation
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(compute_hair_score, image_paths))
    elapsed = time.time() - start_time
    print(f"Hair score computation took {elapsed:.2f} seconds.")

    # get top 5 hairy images
    results = [(score, path) for score, path in results if score > 0]
    results.sort(reverse=True)
    top_hairy = results[:5]

    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    for i, (score, image_path) in enumerate(top_hairy):
        image_filename = os.path.basename(image_path)
        processed_path = os.path.join(processed_dir, image_filename)
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        proc_img = cv2.imread(processed_path)
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original (Hair score: {score})")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(proc_img)
        axes[i, 1].set_title("Hair Removed")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved best hair removal transformation examples to {output_png}")
    plt.show()
    total_elapsed = time.time() - total_start
    print(f"Total script time: {total_elapsed:.2f} seconds.") 
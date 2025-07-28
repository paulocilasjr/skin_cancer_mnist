import os
import pandas as pd

def main():
    # Load metadata
    df = pd.read_csv('image_metadata.csv')

    missing = []
    # Iterate through the image_path column
    for idx, image_name in df['image_path'].items():
        img_path = os.path.join('images_96', image_name)
        if not os.path.isfile(img_path):
            missing.append(image_name)

    # Report
    if missing:
        print("❌ Missing images:")
        for name in missing:
            print(f"  - {name}")
    else:
        print("✅ All images present.")

    print(f"\nSummary: {len(missing)} missing out of {len(df)} total entries.")

if __name__ == '__main__':
    main()

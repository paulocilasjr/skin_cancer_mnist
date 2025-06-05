import zipfile
import os

# Input zip files
zip1_path = 'HAM10000_images_part_1.zip'
zip2_path = 'HAM10000_images_part_2.zip'
output_zip_path = 'HAM10000_images_merged.zip'

# Function to add contents of one zip to another
def merge_zips(zip1, zip2, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zout:
        for zfile in [zip1, zip2]:
            with zipfile.ZipFile(zfile, 'r') as zin:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    # Check for duplicate filenames
                    if item.filename in zout.namelist():
                        print(f"Warning: Duplicate file {item.filename} found. Skipping.")
                        continue
                    zout.writestr(item, data)

# Run the merge
merge_zips(zip1_path, zip2_path, output_zip_path)
print(f"Merged ZIP created at: {output_zip_path}")
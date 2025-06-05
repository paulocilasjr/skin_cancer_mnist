import pandas as pd

df = pd.read_csv('old_processed.csv')
df['image_path'] = df['image_path'].str.replace('HAM10000_all_images/', 'HAM10000_processed/', regex=False)
df.to_csv('old_processed_fixed.csv', index=False)
print('Saved old_processed_fixed.csv with corrected image paths.') 
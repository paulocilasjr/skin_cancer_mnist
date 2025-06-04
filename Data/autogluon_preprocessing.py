import pandas as pd

# Read the original CSV
df = pd.read_csv('Ham10000_Category.csv')

# Split the data
train_df = df[df['split'].isin([0, 1])][['image_id', 'label']]
test_df = df[df['split'] == 2][['image_id', 'label']]

# Save the two new CSV files
train_df.to_csv('HAM10000_hash_train.csv', index=False)
test_df.to_csv('HAM10000_hash_test.csv', index=False)

# Display the first few rows to confirm
print("=== Train Sample ===")
print(train_df.head().to_string(index=False))
print("\n=== Test Sample ===")
print(test_df.head().to_string(index=False))


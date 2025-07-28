import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# === Parameters ===
IMAGE_DIR     = './../images_220'
CSV_FILE      = './../image_metadata_val.csv'
IMG_WIDTH     = 220
IMG_HEIGHT    = 220
EPOCHS        = 150
BATCH_SIZE    = 16
LEARNING_RATE = 0.001

# === Load metadata ===
df = pd.read_csv(CSV_FILE)
print(f"Total rows in metadata: {len(df)}")

# === Split by 'split' column (0=train, 1=val, 2=test) ===
df_train = df[df['split'] == 0].copy()
df_val   = df[df['split'] == 1].copy()
df_test  = df[df['split'] == 2].copy()
print(f"Before filtering → Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# === Filter only valid image paths ===
def filter_existing_images(df_subset):
    def full_path(fn): return os.path.join(IMAGE_DIR, fn)
    mask = df_subset['image_path'].apply(lambda fn: os.path.exists(full_path(fn)))
    if (~mask).any():
        missing = df_subset.loc[~mask, 'image_path'].head(5).tolist()
        print(f"⚠️ Warning: {(~mask).sum()} images not found (e.g., {missing})")
    return df_subset[mask]

df_train = filter_existing_images(df_train)
df_val   = filter_existing_images(df_val)
df_test  = filter_existing_images(df_test)
print(f"After filtering → Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# === Load images and labels ===
def load_images_and_labels(df_subset):
    imgs, labels = [], []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Loading images"):
        path = os.path.join(IMAGE_DIR, row['image_path'])
        try:
            img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            imgs.append(img_to_array(img) / 255.0)
            labels.append(row['label'])
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return np.array(imgs), np.array(labels)

X_train, y_train_raw = load_images_and_labels(df_train)
X_val,   y_val_raw   = load_images_and_labels(df_val)
X_test,  y_test_raw  = load_images_and_labels(df_test)

print(f"Loaded → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError("One of the splits has no images. Check your CSV and paths.")

# === Encode labels ===
le = LabelEncoder()
y_train_i = le.fit_transform(y_train_raw)
y_val_i   = le.transform(y_val_raw)
y_test_i  = le.transform(y_test_raw)

y_train = to_categorical(y_train_i)
y_val   = to_categorical(y_val_i)
y_test  = to_categorical(y_test_i)
NUM_CLASSES = y_train.shape[1]
# Ensure class_names are strings for classification_report
class_names = [str(c) for c in le.classes_]

# === Data Augmentation ===
aug = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# === Build CNN ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D((3,3)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Train ===
history = model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    verbose=1
)

# === Evaluate on Test Set ===
y_pred_prob = model.predict(X_test)
y_pred      = np.argmax(y_pred_prob, axis=1)  # correct variable name
y_true      = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

acc     = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
prec    = precision_score(y_true, y_pred, average='macro')
rec     = recall_score(y_true, y_pred,    average='macro')
f1      = f1_score(y_true, y_pred,        average='macro')
mcc     = matthews_corrcoef(y_true, y_pred)

print(f"\nAccuracy:           {acc:.4f}")
print(f"Balanced Accuracy:  {bal_acc:.4f}")
print(f"Precision (macro):  {prec:.4f}")
print(f"Recall (macro):     {rec:.4f}")
print(f"F1-Score (macro):   {f1:.4f}")
print(f"MCC:                {mcc:.4f}")

# === Plot: Training & Validation Loss/Accuracy ===
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],     label='Train Loss',    linestyle='--')
plt.plot(history.history['val_loss'], label='Val Loss',      linestyle='--')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Training vs. Validation Loss/Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("training_history.png", dpi=300)
plt.close()
print("✅ Saved plot: training_history.png")

# === Plot: Final Evaluation Metrics ===
metrics = {
    "Accuracy":           acc,
    "Balanced Accuracy":  bal_acc,
    "Precision":          prec,
    "Recall":             rec,
    "F1-score":           f1,
    "MCC":                mcc
}
plt.figure(figsize=(10,5))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0,1)
plt.title("Model Performance Metrics on Test Set")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("evaluation_metrics.png", dpi=300)
plt.close()
print("✅ Saved plot: evaluation_metrics.png")


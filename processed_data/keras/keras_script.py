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
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# === Parameters ===
IMAGE_DIR = './../images_96'
CSV_FILE = './../image_metadata.csv'
IMG_WIDTH, IMG_HEIGHT = 96, 96
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# === Load metadata ===
df = pd.read_csv(CSV_FILE)
print(f"Total rows in metadata: {len(df)}")

# === Filter only valid image paths ===
def filter_existing_images(df_subset):
    def full_path(img): return os.path.join(IMAGE_DIR, img)
    valid_rows = df_subset['image_path'].apply(lambda x: os.path.exists(full_path(x)))
    missing_paths = df_subset[~valid_rows]['image_path'].head(5).tolist()
    missing_count = (~valid_rows).sum()
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} images not found (e.g., {missing_paths})")
    return df_subset[valid_rows]

df_train = df[df["split"] == 0]
df_test = df[df["split"] == 2]
print(f"Original splits → Train: {len(df_train)}, Test: {len(df_test)}")

df_train = filter_existing_images(df_train)
df_test = filter_existing_images(df_test)
print(f"After filtering → Train: {len(df_train)}, Test: {len(df_test)}")

# === Load images and labels ===
def load_images_and_labels(df_subset):
    images = []
    labels = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Loading images"):
        img_path = os.path.join(IMAGE_DIR, row['image_path'])
        try:
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img) / 255.0  # Normalize to [0,1]
            images.append(img)
            labels.append(row['label'])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

X_train, y_train_raw = load_images_and_labels(df_train)
X_test, y_test_raw = load_images_and_labels(df_test)

print(f"Loaded → Train images: {len(X_train)}, Test images: {len(X_test)}")

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("No training or testing images were loaded. Check file paths and 'split' column.")

# === Encode labels ===
le = LabelEncoder()
y_train_int = le.fit_transform(y_train_raw)
y_test_int = le.transform(y_test_raw)

y_train_encoded = to_categorical(y_train_int)
y_test_encoded = to_categorical(y_test_int)
NUM_CLASSES = y_train_encoded.shape[1]
class_names = le.classes_

# === Data Augmentation ===
aug = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# === Build CNN ===
model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# === Compile ===
opt = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# === Train ===
history = model.fit(
    aug.flow(X_train, y_train_encoded, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test_encoded),
    epochs=EPOCHS,
    verbose=1
)

# === Evaluate model ===
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test_encoded, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

accuracy = accuracy_score(y_true_classes, y_pred_classes)
balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='macro')
recall = recall_score(y_true_classes, y_pred_classes, average='macro')
f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
mcc = matthews_corrcoef(y_true_classes, y_pred_classes)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-Score (macro): {f1:.4f}")
print(f"MCC: {mcc:.4f}")

# === Plot: Training & Validation Loss/Accuracy ===
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', linestyle='--')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Loss/Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("training_history.png", dpi=300)
plt.close()
print("✅ Saved plot: training_history.png")

# === Plot: Final Evaluation Metrics (Bar Chart) ===
metrics_names = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score", "MCC"]
metrics_values = [accuracy, balanced_acc, precision, recall, f1, mcc]

plt.figure(figsize=(10, 5))
bars = plt.bar(metrics_names, metrics_values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("evaluation_metrics.png", dpi=300)
plt.close()
print("✅ Saved plot: evaluation_metrics.png")

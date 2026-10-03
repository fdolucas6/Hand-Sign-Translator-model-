import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# PATHS
# ============================================================

CSV_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\ad_landmarks.csv"

MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\asl_ad_landmark_model.keras"

# ============================================================
# SETTINGS
# ============================================================

EPOCHS = 8
BATCH_SIZE = 32
SEED = 123

CLASS_NAMES = ["A", "B", "C", "D"]

# ============================================================
# CHECK CSV
# ============================================================

if not os.path.exists(CSV_PATH):
    print("ERROR: CSV file not found:")
    print(CSV_PATH)
    exit()

# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("A-D LANDMARK MODEL TRAINING")
print("========================================")

print("\nLoading:")
print(CSV_PATH)

df = pd.read_csv(CSV_PATH)

X = df.iloc[:, :63].values.astype(np.float32)
y = df["label"].values.astype(np.int32)

print("\nDataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# CLASS COUNTS
# ============================================================

print("\nClass distribution:")

for i, class_name in enumerate(CLASS_NAMES):
    count = np.sum(y == i)
    print(f"{class_name}: {count}")

# ============================================================
# SHUFFLE
# ============================================================

rng = np.random.default_rng(SEED)

indices = np.arange(len(X))
rng.shuffle(indices)

X = X[indices]
y = y[indices]

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split = int(len(X) * 0.80)

X_train = X[:split]
X_val = X[split:]

y_train = y[:split]
y_val = y[split:]

print("\n========================================")
print("TRAIN / VALIDATION SPLIT")
print("========================================")

print("Training samples   :", len(X_train))
print("Validation samples :", len(X_val))

# ============================================================
# BUILD MODEL
# ============================================================

print("\n========================================")
print("BUILDING MODEL")
print("========================================")

model = models.Sequential([

    # 63 landmark values
    layers.Input(shape=(63,)),

    # Hidden layer 1
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.20),

    # Hidden layer 2
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.20),

    # Hidden layer 3
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.10),

    # Output: A B C D
    layers.Dense(4, activation="softmax")
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING - 8 EPOCHS")
print("========================================\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=32,
    shuffle=True,
    verbose=1
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

# ============================================================
# RESULTS
# ============================================================

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

best_train_accuracy = max(history.history["accuracy"])
best_val_accuracy = max(history.history["val_accuracy"])

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(f"\nFinal training accuracy   : {train_accuracy * 100:.2f}%")
print(f"Final validation accuracy : {val_accuracy * 100:.2f}%")

print(f"\nBest training accuracy   : {best_train_accuracy * 100:.2f}%")
print(f"Best validation accuracy : {best_val_accuracy * 100:.2f}%")

print("\nModel saved:")
print(MODEL_PATH)

print("\nClass mapping:")
print("0 = A")
print("1 = B")
print("2 = C")
print("3 = D")

print("\nNEXT STEP:")
print("Run:")
print("python webcam_ad.py")
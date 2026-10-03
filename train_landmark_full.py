import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = r"D:\hand sign chatgpt\hand_sign_project\models"

DATASET_FILE = os.path.join(
    MODELS_DIR,
    "full_landmarks.csv"
)

MODEL_FILE = os.path.join(
    MODELS_DIR,
    "asl_full_landmark_model.keras"
)


# ============================================================
# SETTINGS
# ============================================================

NUM_CLASSES = 27
NUM_FEATURES = 63

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
    "NO_SIGN"
]


# ============================================================
# CHECK DATASET
# ============================================================

print("\n========================================")
print("FULL LANDMARK MODEL TRAINING")
print("========================================")

if not os.path.exists(DATASET_FILE):
    print("\nERROR: Dataset not found:")
    print(DATASET_FILE)
    print("\nRun combine_landmarks.py first.")
    exit()

print("\nDataset found:")
print(DATASET_FILE)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Total samples: {len(df)}")


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

feature_columns = [
    f"feature_{i}"
    for i in range(NUM_FEATURES)
]

X = df[feature_columns].values.astype(np.float32)
y = df["label"].values.astype(np.int32)


print(f"Input shape : {X.shape}")
print(f"Label shape : {y.shape}")


# ============================================================
# CHECK LABELS
# ============================================================

print("\nClass distribution:")

for label in range(NUM_CLASSES):
    count = np.sum(y == label)

    print(
        f"{label:2d} = "
        f"{CLASS_NAMES[label]:8s} : "
        f"{count}"
    )


# ============================================================
# CHECK INVALID LABELS
# ============================================================

invalid_labels = sorted(
    set(y) - set(range(NUM_CLASSES))
)

if invalid_labels:

    print("\nERROR: Invalid labels found:")
    print(invalid_labels)

    exit()

print("\nAll labels are valid.")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train/validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")


# ============================================================
# BUILD CUSTOM MLP
# ============================================================

print("\nBuilding custom neural network...")


model = Sequential([

    Input(shape=(NUM_FEATURES,)),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.20),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.20),

    Dense(64, activation="relu"),
    Dropout(0.10),

    Dense(NUM_CLASSES, activation="softmax")
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n========================================")
print("MODEL ARCHITECTURE")
print("========================================")

model.summary()


# ============================================================
# CALLBACK
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=8,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n========================================")
print("FINAL EVALUATION")
print("========================================")

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(f"\nValidation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_val,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_val,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    y_val,
    predicted_labels,
    labels=list(range(NUM_CLASSES))
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_FILE)


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("\nModel saved to:")
print(MODEL_FILE)

print("\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i:2d} = {name}")

print("\nTotal classes:", NUM_CLASSES)
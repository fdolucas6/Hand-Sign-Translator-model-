import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


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
# CLASSES
# ============================================================

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
    "NO_SIGN"
]


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("FINAL MODEL TEST")
print("========================================")

df = pd.read_csv(DATASET_FILE)

feature_columns = [
    f"feature_{i}"
    for i in range(63)
]

X = df[feature_columns].values.astype(
    np.float32
)

y = df["label"].values.astype(
    np.int32
)


# ============================================================
# SAME VALIDATION SPLIT USED DURING TRAINING
# ============================================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded.")


# ============================================================
# EVALUATE
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n========================================")
print("OVERALL RESULT")
print("========================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test,
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
print("CLASS-WISE PERFORMANCE")
print("========================================")

report = classification_report(
    y_test,
    predicted_labels,
    labels=list(range(27)),
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    y_test,
    predicted_labels,
    labels=list(range(27))
)

print(cm)


# ============================================================
# SIMPLE CLASS ACCURACY
# ============================================================

print("\n========================================")
print("CLASS ACCURACY")
print("========================================")

for i, class_name in enumerate(CLASS_NAMES):

    total = np.sum(y_test == i)

    correct = np.sum(
        (y_test == i) &
        (predicted_labels == i)
    )

    if total > 0:

        class_accuracy = (
            correct / total
        ) * 100

        print(
            f"{class_name:8s} : "
            f"{class_accuracy:.2f}%"
        )

print("\n========================================")
print("TEST COMPLETE")
print("========================================")
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# PATHS
# ============================================================

AD_CSV = r"D:\hand sign chatgpt\hand_sign_project\models\ad_landmarks.csv"

NOTHING_CSV = r"D:\hand sign chatgpt\hand_sign_project\models\nothing_landmarks.csv"

MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\asl_ad_nothing_landmark_model.keras"

# ============================================================
# SETTINGS
# ============================================================

EPOCHS = 8
BATCH_SIZE = 32
SEED = 123

CLASS_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "NO_SIGN"
]

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(AD_CSV):

    print("ERROR: A-D landmark file not found:")
    print(AD_CSV)

    exit()


if not os.path.exists(NOTHING_CSV):

    print("ERROR: NO-SIGN landmark file not found:")
    print(NOTHING_CSV)

    exit()

# ============================================================
# LOAD A-D DATA
# ============================================================

print("\n========================================")
print("LOADING A-D DATA")
print("========================================")

df_ad = pd.read_csv(AD_CSV)

X_ad = df_ad.iloc[:, :63].values.astype(np.float32)

y_ad = df_ad["label"].values.astype(np.int32)

print("A-D samples:", len(X_ad))

# ============================================================
# LOAD NO-SIGN DATA
# ============================================================

print("\n========================================")
print("LOADING NO-SIGN DATA")
print("========================================")

df_nothing = pd.read_csv(NOTHING_CSV)

X_nothing = df_nothing.iloc[:, :63].values.astype(np.float32)

# Every NO-SIGN sample gets label 4
y_nothing = np.full(
    len(X_nothing),
    4,
    dtype=np.int32
)

print("NO-SIGN samples:", len(X_nothing))

# ============================================================
# COMBINE
# ============================================================

X = np.concatenate(
    [X_ad, X_nothing],
    axis=0
)

y = np.concatenate(
    [y_ad, y_nothing],
    axis=0
)

print("\n========================================")
print("COMBINED DATASET")
print("========================================")

print("Total samples:", len(X))
print("Features:", X.shape[1])

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(y == i)

    print(
        f"{i} = {class_name}: {count}"
    )

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
print("DATASET SPLIT")
print("========================================")

print("Training samples:", len(X_train))

print("Validation samples:", len(X_val))

# ============================================================
# BUILD MODEL
# ============================================================

print("\n========================================")
print("BUILDING 5-CLASS MODEL")
print("========================================")

model = models.Sequential([

    # 63 MediaPipe landmark values
    layers.Input(shape=(63,)),

    # Hidden layer 1
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.20),

    # Hidden layer 2
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.20),

    # Hidden layer 3
    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.10),

    # Output
    # A / B / C / D / NO_SIGN
    layers.Dense(
        5,
        activation="softmax"
    )
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
print("STARTING TRAINING")
print("8 EPOCHS")
print("========================================\n")

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

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

final_train_accuracy = history.history["accuracy"][-1]

final_val_accuracy = history.history["val_accuracy"][-1]

best_train_accuracy = max(
    history.history["accuracy"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"\nFinal training accuracy   : "
    f"{final_train_accuracy * 100:.2f}%"
)

print(
    f"Final validation accuracy : "
    f"{final_val_accuracy * 100:.2f}%"
)

print(
    f"\nBest training accuracy   : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best validation accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print("\nModel saved to:")

print(MODEL_PATH)

print("\nClass mapping:")

print("0 = A")
print("1 = B")
print("2 = C")
print("3 = D")
print("4 = NO_SIGN")

print("\nNext:")
print("We will test the 5-class model with the webcam.")
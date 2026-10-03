import os
import pandas as pd

MODELS_DIR = r"D:\hand sign chatgpt\hand_sign_project\models"

AD_FILE = os.path.join(MODELS_DIR, "ad_landmarks.csv")
EZ_FILE = os.path.join(MODELS_DIR, "ez_landmarks.csv")
NOTHING_FILE = os.path.join(MODELS_DIR, "nothing_landmarks.csv")

OUTPUT_FILE = os.path.join(MODELS_DIR, "full_landmarks.csv")


print("\n========================================")
print("COMBINING LANDMARK DATASETS")
print("========================================")


# --------------------------------------------------
# Check files
# --------------------------------------------------

for file in [AD_FILE, EZ_FILE, NOTHING_FILE]:
    if not os.path.exists(file):
        print("\nERROR: File not found:")
        print(file)
        exit()

print("\nAll required files found.")


# --------------------------------------------------
# Load A-D
# --------------------------------------------------

print("\nLoading A-D dataset...")

df_ad = pd.read_csv(AD_FILE)

print(f"A-D samples: {len(df_ad)}")


# A-D already have correct labels:
# 0=A
# 1=B
# 2=C
# 3=D


# --------------------------------------------------
# Load E-Z
# --------------------------------------------------

print("\nLoading E-Z dataset...")

df_ez = pd.read_csv(EZ_FILE)

print(f"E-Z samples before label conversion: {len(df_ez)}")


# E-Z currently has:
#
# 0 = E
# 1 = F
# 2 = G
# ...
# 21 = Z
#
# Convert them to:
#
# 4 = E
# 5 = F
# ...
# 25 = Z

df_ez["label"] = df_ez["label"] + 4


# --------------------------------------------------
# Load NO_SIGN
# --------------------------------------------------

print("\nLoading NO_SIGN dataset...")

df_nothing = pd.read_csv(NOTHING_FILE)

print(f"NO_SIGN samples: {len(df_nothing)}")


# Current NO_SIGN label is 4.
# Change it to 26 for the final model.

df_nothing["label"] = 26


# --------------------------------------------------
# Combine all datasets
# --------------------------------------------------

print("\nCombining datasets...")

df_full = pd.concat(
    [df_ad, df_ez, df_nothing],
    ignore_index=True
)


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

df_full = df_full.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save
# --------------------------------------------------

df_full.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Display information
# --------------------------------------------------

print("\n========================================")
print("COMBINATION COMPLETE")
print("========================================")

print(f"\nTotal samples: {len(df_full)}")

print("\nClass distribution:")

class_names = [
    "A", "B", "C", "D",
    "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "NO_SIGN"
]

for label in range(27):
    count = (df_full["label"] == label).sum()
    print(f"{label:2d} = {class_names[label]:8s} : {count}")


print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFinal classes:")
print("A-Z + NO_SIGN = 27 classes")
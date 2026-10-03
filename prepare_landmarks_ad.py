import os
import csv
import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# PATHS
# ============================================================

SOURCE_DATASET = r"D:\hand sign chatgpt\asl_alphabet_train\asl_alphabet_train"

OUTPUT_FILE = r"D:\hand sign chatgpt\hand_sign_project\models\ad_landmarks.csv"

HAND_MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\hand_landmarker.task"


# ============================================================
# A-D CLASSES ONLY
# ============================================================

CLASSES = ["A", "B", "C", "D"]


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(SOURCE_DATASET):
    print("ERROR: Dataset folder not found:")
    print(SOURCE_DATASET)
    exit()

if not os.path.exists(HAND_MODEL_PATH):
    print("ERROR: MediaPipe model not found:")
    print(HAND_MODEL_PATH)
    exit()


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)


detector = HandLandmarker.create_from_options(
    options
)


# ============================================================
# NORMALIZE 21 LANDMARKS
# ============================================================

def normalize_landmarks(landmarks):

    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in landmarks
        ],
        dtype=np.float32
    )

    # Landmark 0 = wrist
    wrist = points[0].copy()

    # Move wrist to origin
    points = points - wrist

    # Use middle finger MCP as scale reference
    palm_scale = np.linalg.norm(points[9])

    if palm_scale < 1e-6:
        palm_scale = 1.0

    points = points / palm_scale

    # 21 landmarks × 3 coordinates = 63 values
    return points.flatten()


# ============================================================
# PREPARE DATASET
# ============================================================

rows = []

total_images = 0
successful_images = 0
failed_images = 0


print("\n========================================")
print("A-D LANDMARK DATASET PREPARATION")
print("========================================")

print("\nDataset:")
print(SOURCE_DATASET)

print("\nClasses:")
print(CLASSES)


# ============================================================
# PROCESS A-D
# ============================================================

for label_index, class_name in enumerate(CLASSES):

    class_folder = os.path.join(
        SOURCE_DATASET,
        class_name
    )

    if not os.path.exists(class_folder):

        print(
            f"\nERROR: Folder not found: {class_folder}"
        )

        detector.close()
        exit()

    image_files = [
        filename
        for filename in os.listdir(class_folder)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(
        f"\nProcessing {class_name}: "
        f"{len(image_files)} images"
    )

    for filename in image_files:

        total_images += 1

        image_path = os.path.join(
            class_folder,
            filename
        )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:

            failed_images += 1
            continue

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = detector.detect(
            mp_image
        )

        if not result.hand_landmarks:
            continue

        # ----------------------------------------------------
        # FIRST HAND
        # ----------------------------------------------------

        landmarks = result.hand_landmarks[0]

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        features = normalize_landmarks(
            landmarks
        )

        # ----------------------------------------------------
        # SAVE FEATURE + LABEL
        # ----------------------------------------------------

        row = features.tolist()

        row.append(label_index)

        rows.append(row)

        successful_images += 1


# ============================================================
# SAVE CSV
# ============================================================

header = [
    f"feature_{i}"
    for i in range(63)
]

header.append("label")


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(header)

    writer.writerows(rows)


# ============================================================
# CLOSE MEDIAPIPE
# ============================================================

detector.close()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n========================================")
print("DATASET PREPARATION COMPLETE")
print("========================================")

print(
    f"Total images processed : {total_images}"
)

print(
    f"Successful landmarks   : {successful_images}"
)

print(
    f"Images skipped         : "
    f"{total_images - successful_images}"
)

print(
    f"Failed images          : {failed_images}"
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nClass labels:")
print("0 = A")
print("1 = B")
print("2 = C")
print("3 = D")
import os
import csv
import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# PATHS
# ============================================================

OUTPUT_FILE = r"D:\hand sign chatgpt\hand_sign_project\models\nothing_landmarks.csv"

HAND_MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\hand_landmarker.task"

# ============================================================
# SETTINGS
# ============================================================

TARGET_SAMPLES = 3000

# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(HAND_MODEL_PATH):
    print("ERROR: MediaPipe model not found:")
    print(HAND_MODEL_PATH)
    exit()

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ============================================================
# MEDIAPIPE
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

detector = HandLandmarker.create_from_options(options)

# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(landmarks):

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    # Wrist = origin
    wrist = points[0].copy()
    points = points - wrist

    # Same normalization used for A-D
    palm_scale = np.linalg.norm(points[9])

    if palm_scale < 1e-6:
        palm_scale = 1.0

    points = points / palm_scale

    # 21 × 3 = 63 values
    return points.flatten()

# ============================================================
# CSV HEADER
# ============================================================

header = [f"feature_{i}" for i in range(63)]

# Label 4 = NO SIGN
header.append("label")

# ============================================================
# OPEN CSV
# ============================================================

file = open(
    OUTPUT_FILE,
    "w",
    newline=""
)

writer = csv.writer(file)

writer.writerow(header)

# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    file.close()
    detector.close()

    exit()

print("\n========================================")
print("NORMAL HAND / NO-SIGN DATA COLLECTION")
print("========================================")

print("\nTarget samples:", TARGET_SAMPLES)

print("\nIMPORTANT:")
print("DO NOT make A, B, C, or D.")
print("Show your normal/relaxed hand.")
print("Move your hand into different normal positions.")

print("\nPress Q to quit.")
print("Press S to start/stop collection.\n")

# ============================================================
# VARIABLES
# ============================================================

sample_count = 0
collecting = False

# ============================================================
# WEBCAM LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    hand_detected = False

    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand_detected = True

        landmarks = result.hand_landmarks[0]

        # Draw landmarks
        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # ====================================================
        # COLLECT NORMAL HAND
        # ====================================================

        if collecting and sample_count < TARGET_SAMPLES:

            features = normalize_landmarks(landmarks)

            row = features.tolist()

            # Label 4 = NO SIGN
            row.append(4)

            writer.writerow(row)

            sample_count += 1

    # ========================================================
    # STATUS
    # ========================================================

    if collecting:

        status = "COLLECTING"

    else:

        status = "PAUSED"

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{TARGET_SAMPLES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if hand_detected:

        cv2.putText(
            frame,
            "Hand detected",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        "S = Start/Stop | Q = Quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Collect NO-SIGN Hand Landmarks",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        collecting = not collecting

        if collecting:

            print("\nCollection STARTED")

        else:

            print("\nCollection PAUSED")

    if key == ord("q"):
        break

    # ========================================================
    # TARGET REACHED
    # ========================================================

    if sample_count >= TARGET_SAMPLES:

        print("\nTarget samples reached!")

        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

file.close()
detector.close()

print("\n========================================")
print("COLLECTION COMPLETE")
print("========================================")

print("Samples collected:", sample_count)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nLabel:")
print("4 = NO SIGN")
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\asl_ad_nothing_landmark_model.keras"

HAND_MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\hand_landmarker.task"

# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "NO SIGN"
]

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading 5-class landmark model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

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
# NORMALIZATION
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

    # Palm scale
    palm_scale = np.linalg.norm(
        points[9]
    )

    if palm_scale < 1e-6:
        palm_scale = 1.0

    points = points / palm_scale

    # 21 × 3 = 63
    return points.flatten()

# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    exit()

print("\n========================================")
print("A B C D / NO SIGN WEBCAM TEST")
print("========================================")

print("\nPress Q to quit.")

# ============================================================
# LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read webcam.")

        break

    # Mirror
    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    # BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb
    )

    # ========================================================
    # MEDIAPIPE DETECTION
    # ========================================================

    result = detector.detect(
        mp_image
    )

    prediction_text = "NO SIGN"

    confidence = 0.0

    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        for landmark in landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(

                frame,

                (x, y),

                5,

                (0, 255, 0),

                -1
            )

        # ----------------------------------------------------
        # Extract 63 values
        # ----------------------------------------------------

        features = normalize_landmarks(
            landmarks
        )

        model_input = np.expand_dims(
            features,
            axis=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(

            model_input,

            verbose=0
        )[0]

        predicted_index = np.argmax(
            predictions
        )

        confidence = (
            predictions[predicted_index]
            * 100
        )

        prediction_text = CLASS_NAMES[
            predicted_index
        ]

    # ========================================================
    # NO HAND
    # ========================================================

    else:

        prediction_text = "NO SIGN"

        confidence = 100.0

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(

        frame,

        f"Prediction: {prediction_text}",

        (20, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.1,

        (0, 255, 0),

        3
    )

    cv2.putText(

        frame,

        f"Confidence: {confidence:.2f}%",

        (20, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )

    cv2.putText(

        frame,

        "A / B / C / D / NO SIGN",

        (20, height - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )

    cv2.imshow(

        "ASL Landmark Translator",

        frame
    )

    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("\nWebcam test finished.")
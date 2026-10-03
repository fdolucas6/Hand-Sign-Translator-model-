import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\asl_full_landmark_model.keras"

HAND_MODEL_PATH = r"D:\hand sign chatgpt\hand_sign_project\models\hand_landmarker.task"


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
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

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
            [landmark.x, landmark.y, landmark.z]
            for landmark in landmarks
        ],
        dtype=np.float32
    )

    # Move wrist to origin
    wrist = points[0].copy()

    points = points - wrist

    # Normalize hand size
    palm_scale = np.linalg.norm(points[9])

    if palm_scale < 1e-6:
        palm_scale = 1.0

    points = points / palm_scale

    return points.flatten()


# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_landmarks(frame, landmarks):

    h, w, _ = frame.shape

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for start, end in connections:

        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)

        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

    for landmark in landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()
    exit()


print("\n========================================")
print("FULL HAND SIGN TRANSLATOR")
print("========================================")
print("\nA-Z + NO_SIGN")
print("\nPress Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # Detect hand
    result = detector.detect(mp_image)


    # Default
    predicted_text = "NO SIGN"
    confidence = 0.0


    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]


        # Draw landmarks
        draw_landmarks(
            frame,
            landmarks
        )


        # Normalize
        features = normalize_landmarks(
            landmarks
        )


        # Reshape for model
        input_data = features.reshape(
            1,
            63
        )


        # Prediction
        prediction = model.predict(
            input_data,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(prediction)
        )


        confidence = float(
            prediction[predicted_index]
        )


        predicted_text = CLASS_NAMES[
            predicted_index
        ]


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (500, 100),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Prediction: {predicted_text}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Hand Sign Translator",
        frame
    )


    # Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("\nWebcam closed.")
print("Program finished.")
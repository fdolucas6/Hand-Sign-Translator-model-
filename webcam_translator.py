import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
from collections import Counter


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
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.70

STABLE_FRAMES = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# TEXT TO SPEECH
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)


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
            [landmark.x, landmark.y, landmark.z]
            for landmark in landmarks
        ],
        dtype=np.float32
    )

    # Wrist becomes origin
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
# VARIABLES
# ============================================================

text_buffer = ""

prediction_history = []

previous_stable_prediction = "NO_SIGN"

# Prevent the same sign from being added repeatedly
letter_already_added = False


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    exit()


print("\n========================================")
print("HAND SIGN TRANSLATOR")
print("========================================")

print("\nClasses:")
print("A-Z + NO_SIGN")

print("\nControls:")
print("SPACE     = Add space")
print("BACKSPACE = Delete character")
print("C         = Clear text")
print("S         = Speak text")
print("Q         = Quit")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break


    # Mirror image
    frame = cv2.flip(frame, 1)


    # Convert to RGB
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


    current_prediction = "NO_SIGN"
    confidence = 0.0


    # ========================================================
    # HAND DETECTED
    # ========================================================

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


        # Model input
        input_data = features.reshape(
            1,
            63
        )


        # Predict
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


        current_prediction = CLASS_NAMES[
            predicted_index
        ]


        # Low confidence = NO_SIGN
        if confidence < CONFIDENCE_THRESHOLD:

            current_prediction = "NO_SIGN"


    else:

        current_prediction = "NO_SIGN"


    # ========================================================
    # STABLE PREDICTION
    # ========================================================

    prediction_history.append(
        current_prediction
    )


    if len(prediction_history) > STABLE_FRAMES:

        prediction_history.pop(0)


    stable_prediction = "NO_SIGN"


    if len(prediction_history) == STABLE_FRAMES:

        counts = Counter(
            prediction_history
        )

        most_common = counts.most_common(1)[0]

        candidate = most_common[0]
        candidate_count = most_common[1]


        if candidate_count >= STABLE_FRAMES - 1:

            stable_prediction = candidate


    # ========================================================
    # ADD LETTER
    # ========================================================

    if stable_prediction == "NO_SIGN":

        # Reset so a new sign can be entered
        letter_already_added = False


    elif stable_prediction != "NO_SIGN":

        if not letter_already_added:

            text_buffer += stable_prediction

            letter_already_added = True

            print(
                "Added:",
                stable_prediction
            )


    previous_stable_prediction = stable_prediction


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (900, 155),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Sign: {stable_prediction}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    display_text = text_buffer

    if len(display_text) > 55:

        display_text = display_text[-55:]


    cv2.putText(
        frame,
        f"Text: {display_text}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "A-Z | NO_SIGN",
        (650, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # Show webcam
    cv2.imshow(
        "Hand Sign Translator",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord("q"):

        break


    # Clear
    elif key == ord("c"):

        text_buffer = ""

        print("Text cleared.")


    # Speak
    elif key == ord("s"):

        if text_buffer.strip():

            print(
                "Speaking:",
                text_buffer
            )

            engine.say(
                text_buffer
            )

            engine.runAndWait()


    # Space
    elif key == 32:

        text_buffer += " "


    # Backspace
    elif key == 8:

        text_buffer = text_buffer[:-1]


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("\n========================================")
print("TRANSLATOR CLOSED")
print("========================================")
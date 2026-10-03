import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\hand sign chatgpt\hand_sign_project"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "asl_full_landmark_model.keras"
)

TASK_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 5

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "NO_SIGN"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = load_model(MODEL_PATH)

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
        model_asset_path=TASK_MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# TEXT TO SPEECH
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)


def speak_text():
    text = text_var.get().strip()

    if not text:
        return

    def speak():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=speak, daemon=True).start()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_landmarks(landmarks):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Wrist
    wrist = points[0].copy()

    # Translate wrist to origin
    points = points - wrist

    # Palm scale
    palm_scale = np.linalg.norm(points[9])

    if palm_scale < 1e-6:
        return None

    # Scale normalization
    points = points / palm_scale

    # Flatten 21 x 3 = 63 values
    return points.flatten()


# ============================================================
# GLOBAL VARIABLES
# ============================================================

cap = cv2.VideoCapture(0)

running = True

text_buffer = ""

last_prediction = "NO_SIGN"
stable_count = 0

letter_ready = True


# ============================================================
# TKINTER WINDOW
# ============================================================

root = tk.Tk()

root.title("Hand Sign Translator")

root.geometry("1200x750")

root.configure(bg="#101820")

root.resizable(False, False)


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

style.theme_use("clam")

style.configure(
    "Title.TLabel",
    background="#101820",
    foreground="white",
    font=("Arial", 26, "bold")
)

style.configure(
    "Normal.TLabel",
    background="#101820",
    foreground="white",
    font=("Arial", 14)
)

style.configure(
    "Prediction.TLabel",
    background="#101820",
    foreground="#00ff99",
    font=("Arial", 42, "bold")
)

style.configure(
    "Confidence.TLabel",
    background="#101820",
    foreground="white",
    font=("Arial", 16)
)


# ============================================================
# TITLE
# ============================================================

title = ttk.Label(
    root,
    text="HAND SIGN TRANSLATOR",
    style="Title.TLabel"
)

title.pack(pady=15)


# ============================================================
# MAIN FRAME
# ============================================================

main_frame = tk.Frame(
    root,
    bg="#101820"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)


# ============================================================
# LEFT CAMERA FRAME
# ============================================================

camera_frame = tk.Frame(
    main_frame,
    bg="#1b2630",
    width=720,
    height=540
)

camera_frame.pack(
    side="left",
    padx=10
)

camera_frame.pack_propagate(False)


camera_title = tk.Label(
    camera_frame,
    text="LIVE CAMERA",
    bg="#1b2630",
    fg="white",
    font=("Arial", 18, "bold")
)

camera_title.pack(pady=10)


camera_label = tk.Label(
    camera_frame,
    bg="black"
)

camera_label.pack(
    padx=10,
    pady=5
)


# ============================================================
# RIGHT INFORMATION FRAME
# ============================================================

info_frame = tk.Frame(
    main_frame,
    bg="#1b2630",
    width=400,
    height=540
)

info_frame.pack(
    side="right",
    padx=10
)

info_frame.pack_propagate(False)


# ============================================================
# DETECTED SIGN
# ============================================================

detected_title = tk.Label(
    info_frame,
    text="DETECTED SIGN",
    bg="#1b2630",
    fg="white",
    font=("Arial", 16, "bold")
)

detected_title.pack(pady=(25, 5))


prediction_var = tk.StringVar(
    value="NO SIGN"
)

prediction_label = tk.Label(
    info_frame,
    textvariable=prediction_var,
    bg="#1b2630",
    fg="#00ff99",
    font=("Arial", 44, "bold")
)

prediction_label.pack(pady=10)


# ============================================================
# CONFIDENCE
# ============================================================

confidence_var = tk.StringVar(
    value="Confidence: 0%"
)

confidence_label = tk.Label(
    info_frame,
    textvariable=confidence_var,
    bg="#1b2630",
    fg="white",
    font=("Arial", 15)
)

confidence_label.pack(pady=5)


# ============================================================
# TEXT
# ============================================================

text_title = tk.Label(
    info_frame,
    text="TRANSLATED TEXT",
    bg="#1b2630",
    fg="white",
    font=("Arial", 16, "bold")
)

text_title.pack(pady=(35, 8))


text_var = tk.StringVar()

text_entry = tk.Entry(
    info_frame,
    textvariable=text_var,
    bg="#ffffff",
    fg="#111111",
    font=("Arial", 18),
    width=25,
    justify="center"
)

text_entry.pack(
    padx=20,
    ipady=12
)


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    info_frame,
    bg="#1b2630"
)

button_frame.pack(
    pady=30
)


# ============================================================
# SPEAK
# ============================================================

speak_button = tk.Button(
    button_frame,
    text="🔊 SPEAK",
    command=speak_text,
    bg="#2563eb",
    fg="white",
    font=("Arial", 13, "bold"),
    width=12,
    height=2,
    relief="flat",
    cursor="hand2"
)

speak_button.grid(
    row=0,
    column=0,
    padx=5
)


# ============================================================
# CLEAR
# ============================================================

def clear_text():
    text_var.set("")


clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    command=clear_text,
    bg="#444444",
    fg="white",
    font=("Arial", 13, "bold"),
    width=12,
    height=2,
    relief="flat",
    cursor="hand2"
)

clear_button.grid(
    row=0,
    column=1,
    padx=5
)


# ============================================================
# STATUS
# ============================================================

status_var = tk.StringVar(
    value="System ready"
)

status_label = tk.Label(
    info_frame,
    textvariable=status_var,
    bg="#1b2630",
    fg="#bbbbbb",
    font=("Arial", 12)
)

status_label.pack(
    pady=10
)


# ============================================================
# ADD LETTER
# ============================================================

def add_letter(letter):

    global letter_ready

    if letter_ready:

        current = text_var.get()

        text_var.set(
            current + letter
        )

        letter_ready = False


# ============================================================
# CAMERA PROCESSING
# ============================================================

def update_camera():

    global running
    global last_prediction
    global stable_count
    global letter_ready

    if not running:
        return

    ret, frame = cap.read()

    if not ret:

        status_var.set(
            "Unable to access camera"
        )

        root.after(
            30,
            update_camera
        )

        return

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect landmarks
    result = landmarker.detect(
        mp_image
    )

    prediction = "NO_SIGN"
    confidence = 0.0

    # --------------------------------------------------------
    # HAND DETECTED
    # --------------------------------------------------------

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        features = normalize_landmarks(
            landmarks
        )

        if features is not None:

            input_data = np.expand_dims(
                features,
                axis=0
            )

            probabilities = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[predicted_index]
            )

            prediction = CLASS_NAMES[
                predicted_index
            ]

            # Confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:

                prediction = "NO_SIGN"

    else:

        prediction = "NO_SIGN"


    # ========================================================
    # STABLE PREDICTION
    # ========================================================

    if prediction == last_prediction:

        stable_count += 1

    else:

        stable_count = 0

        last_prediction = prediction


    # Add letter after stable prediction
    if (
        prediction != "NO_SIGN"
        and stable_count >= STABLE_FRAMES
    ):

        add_letter(prediction)

        stable_count = 0


    # NO SIGN allows next letter
    if prediction == "NO_SIGN":

        letter_ready = True


    # ========================================================
    # UPDATE GUI
    # ========================================================

    if prediction == "NO_SIGN":

        prediction_var.set(
            "NO SIGN"
        )

        prediction_label.configure(
            fg="#aaaaaa"
        )

        status_var.set(
            "Show an ASL letter"
        )

    else:

        prediction_var.set(
            prediction
        )

        prediction_label.configure(
            fg="#00ff99"
        )

        status_var.set(
            "Sign detected"
        )


    confidence_var.set(
        f"Confidence: {confidence * 100:.1f}%"
    )


    # ========================================================
    # DRAW CAMERA INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        prediction,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 100),
        3
    )

    cv2.putText(
        frame,
        f"{confidence * 100:.1f}%",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )

    image = image.resize(
        (680, 450)
    )

    photo = ImageTk.PhotoImage(
        image=image
    )

    camera_label.configure(
        image=photo
    )

    camera_label.image = photo


    root.after(
        15,
        update_camera
    )


# ============================================================
# EXIT
# ============================================================

def close_application():

    global running

    running = False

    cap.release()

    landmarker.close()

    root.destroy()


exit_button = tk.Button(
    root,
    text="EXIT",
    command=close_application,
    bg="#b91c1c",
    fg="white",
    font=("Arial", 13, "bold"),
    width=15,
    height=2,
    relief="flat",
    cursor="hand2"
)

exit_button.pack(
    pady=15
)


# ============================================================
# START
# ============================================================

root.protocol(
    "WM_DELETE_WINDOW",
    close_application
)

update_camera()

root.mainloop()
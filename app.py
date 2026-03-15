import queue
import threading
import time
from collections import deque, Counter
import cv2
import numpy as np
import sys
import pygame
from keras.models import load_model
import os
from io import BytesIO
from gtts import gTTS
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

model = load_model('models/CNN_LSTM_model.keras')
POSE_MODEL = 'process_data/pose_landmarker_full.task'
HAND_MODEL = 'process_data/hand_landmarker.task'

tts_queue = queue.Queue()
pygame.init()
pygame.mixer.init()

# text-to-speech
def speak(text, language='en'):
    mp3_fo = BytesIO()
    tts = gTTS(text, lang=language)
    tts.write_to_fp(mp3_fo)
    mp3_fo.seek(0)
    return mp3_fo


def tts_worker():
    while True:
        text = tts_queue.get() #waits until a new word is available
        if text:
            try:
                sound = speak(text)
                pygame.mixer.music.load(sound, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in TTS: {e}")
        tts_queue.task_done()

prediction_buffer = deque(maxlen=10) #holds last 10 predictions
sequence_buffer = deque(maxlen=30)
stable_prediction = deque(maxlen=20)

final_text = ""

DATA_DIR = 'processed_data'
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] # classes

base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL)
base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL)

pose_options = vision.PoseLandmarkerOptions(base_options=base_options_pose, running_mode=vision.RunningMode.IMAGE)
hand_options = vision.HandLandmarkerOptions(base_options=base_options_hand, running_mode=vision.RunningMode.IMAGE, num_hands=2)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

def preprocess_hand(crop):
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    resized = cv2.resize(img_thresh, (128, 128))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 128, 128, 1)
    return reshaped

#crops out the hand from the image
def get_square_box(landmarks, shape, padding=20):
    h, w, _ = shape
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]

    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, h)

    box_w = x_max - x_min
    box_h = y_max - y_min
    box_size = max(box_w, box_h)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    x1 = max(cx - box_size // 2, 0)
    y1 = max(cy - box_size // 2, 0)
    x2 = min(cx + box_size // 2, w)
    y2 = min(cy + box_size // 2, h)

    return x1, y1, x2, y2

cap = cv2.VideoCapture(0) #webcam initialized

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))

    pose_res = pose_landmarker.detect(mp_image)
    hand_res = hand_landmarker.detect(mp_image)

    # --- 1. Extract Pose (Stay consistent with training normalization!) ---
    pose = np.zeros(132)

    # --- 2. Extract Hands ---
    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)

    if hand_res.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_res.hand_landmarks):
            wrist = hand_landmarks[0]
            # Normalize relative to wrist
            normalized = []
            for lm in hand_landmarks:
                normalized.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            label = hand_res.handedness[i][0].category_name
            if label == 'Left':
                lh = np.array(normalized).flatten()
            else:
                rh = np.array(normalized).flatten()

            # Drawing logic stays here (Visual only)
            x1, y1, x2, y2 = get_square_box(hand_landmarks, frame.shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- 3. Sequence & Prediction (OUTSIDE the hand loop) ---
    current_frame_features = np.concatenate([pose, lh, rh])
    sequence_buffer.append(current_frame_features)

    if len(sequence_buffer) == 30:
        input_data = np.expand_dims(list(sequence_buffer), axis=0)
        predictions = model.predict(input_data, verbose=0)
        pred_idx = np.argmax(predictions)
        confidence = predictions[0][pred_idx]
        predicted_class = class_names[pred_idx]

        # Display the prediction globally on the frame
        cv2.putText(frame, f"PREDICTION: {predicted_class} ({confidence * 100:.1f}%)",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if pose_res.pose_landmarks:

        drawing_utils.draw_landmarks(

            frame, pose_res.pose_landmarks[0], vision.PoseLandmarksConnections.POSE_LANDMARKS,

            drawing_styles.get_default_pose_landmarks_style()
        )

    if hand_res.hand_landmarks:

        for hand_lms in hand_res.hand_landmarks:

            drawing_utils.draw_landmarks(

                frame, hand_lms, vision.HandLandmarksConnections.HAND_CONNECTIONS,

                drawing_styles.get_default_hand_landmarks_style(),

                drawing_styles.get_default_hand_connections_style()
            )

    cv2.imshow("ASL Interpreter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


cap.release()
cv2.destroyAllWindows()
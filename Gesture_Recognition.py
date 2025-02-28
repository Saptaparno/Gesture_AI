import cv2
import numpy as np
import mediapipe as mp
import pyttsx3  # Import text-to-speech
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load trained model
model = load_model("../models/gesture_model.h5")

# Load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load("./data/label_classes.npy", allow_pickle=True)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
last_spoken = None  # Track last spoken word to avoid repeated speech

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmarks = landmarks / np.max(landmarks)  # Normalize
            landmarks = landmarks.reshape(1, 21, 3)  # Reshape for model

            # Predict gesture
            prediction = model.predict(landmarks)
            max_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Confidence threshold
            threshold = 0.6
            if confidence > threshold:
                gesture = encoder.inverse_transform([max_index])[0]
            else:
                gesture = "Unknown"

            # **Text-to-Speech Output**
            if gesture != last_spoken:  # Avoid repeating the same word
                engine.say(gesture)  # Speak the gesture
                engine.runAndWait()
                last_spoken = gesture  # Update last spoken word

            # Display gesture on frame
            cv2.putText(frame, f"{gesture} ({confidence:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

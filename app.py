import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow.lite as tflite
from flask import Flask, Response, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load TensorFlow Lite model
MODEL_PATH = "gesture_model.tflite"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture video stream
cap = cv2.VideoCapture(0)

@app.route("/")
def home():
    return "Gesture Recognition API is Running!"

def process_frame():
    """Capture and process frames for gesture recognition."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                landmarks = np.array(landmarks, dtype=np.float32).reshape(1, 21, 3)

                # Run inference
                interpreter.set_tensor(input_details[0]['index'], landmarks)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                # Get the highest probability class
                predicted_class = np.argmax(prediction)
                
                # Display prediction on screen
                cv2.putText(frame, f"Gesture: {predicted_class}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode and return frame
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(process_frame(), mimetype="multipa

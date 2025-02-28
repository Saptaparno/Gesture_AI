import os
import cv2
import numpy as np
import tensorflow.lite as tflite
import mediapipe as mp
from flask import Flask, jsonify, Response

# Initialize Flask App
app = Flask(__name__)

# Ensure model exists
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_model.tflite")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# Load TFLite Model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load gesture labels
LABELS_PATH = os.path.join(os.path.dirname(__file__), "data", "labels.npy")
if os.path.exists(LABELS_PATH):
    labels = np.load(LABELS_PATH, allow_pickle=True)
else:
    labels = []

# Initialize Mediapipe for Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start Camera
cap = cv2.VideoCapture(0)


def preprocess_landmarks(landmarks):
    """ Normalize landmarks for model input """
    if landmarks:
        landmarks = np.array(landmarks).flatten()
        return landmarks.reshape(1, -1).astype(np.float32)
    return None


def predict_gesture(landmarks):
    """ Predict gesture from landmarks using TFLite model """
    if landmarks is None:
        return "Unknown"

    interpreter.set_tensor(input_details[0]['index'], landmarks)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    if np.max(prediction) < 0.6:  # Confidence threshold
        return "Unknown"

    return labels[np.argmax(prediction)]


def detect_hand_gesture(frame):
    """ Detect hand gesture using Mediapipe """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            return preprocess_landmarks(landmarks)
    return None


@app.route("/detect_gesture", methods=["GET"])
def detect_gesture():
    """ API Endpoint to detect hand gestures """
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Camera not working"}), 500

    # Detect hand gesture
    hand_landmarks = detect_hand_gesture(frame)
    gesture = predict_gesture(hand_landmarks)

    return jsonify({"gesture": gesture})


@app.route("/video_feed")
def video_feed():
    """ Live video feed with gesture recognition (for debugging) """
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gesture = detect_gesture().json.get("gesture", "Unknown")
            cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("✅ Starting Gesture Recognition Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)

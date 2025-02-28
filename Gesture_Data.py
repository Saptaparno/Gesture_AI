import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

# Ensure 'Data' folder exists
os.makedirs("./Data", exist_ok=True)
print("✅ 'Data' folder is ready.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

gesture_data = []
labels = []
gesture_count = 0

print("Press 's' to save a sample, 'q' to quit")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                cv2.putText(frame, "Press 's' to save this gesture", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Collect Gesture Data', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Check for 'q' first
            print("Exiting...")
            break
        elif key == ord('s'):  # Check for 's' after
            gesture_name = input("Enter gesture label: ")
            gesture_data.append(landmarks)
            labels.append(gesture_name)
            gesture_count += 1
            print(f"✅ Saved gesture {gesture_count}: {gesture_name}")
            print(f"Current data size: {len(gesture_data)}")

except Exception as e:
    print(f"❌ An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Save data
    if gesture_data:
        print("Saving data...")
        # Check if files already exist
        gestures_file = "./Data/gestures.npy"
        labels_file = "./Data/labels.npy"

        if os.path.exists(gestures_file) and os.path.exists(labels_file):
            # Load existing data
            existing_gestures = np.load(gestures_file, allow_pickle=True)
            existing_labels = np.load(labels_file, allow_pickle=True)

            # Append new data
            updated_gestures = np.append(existing_gestures, gesture_data, axis=0)
            updated_labels = np.append(existing_labels, labels, axis=0)

            # Save updated data
            np.save(gestures_file, updated_gestures)
            np.save(labels_file, updated_labels)
        else:
            # Save new data
            np.save(gestures_file, np.array(gesture_data))
            np.save(labels_file, np.array(labels))

        print("✅ Data saved successfully!")
    else:
        print("❌ No data to save.")
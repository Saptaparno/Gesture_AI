import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import LabelEncoder

# Load dataset
X_train = np.load("./Data/gestures.npy", allow_pickle=True)
y_train = np.load("./Data/labels.npy", allow_pickle=True)

# Flatten input shape dynamically
input_shape = X_train[0].shape
X_train = np.array([x.flatten() for x in X_train])

# Convert labels to numbers
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# Define model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Dynamic input shape
    Dense(64, activation='relu'),
    Dense(len(set(y_train)), activation='softmax')  # Dynamic output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=4)  # Small batch size for few samples

# Save model & label encoder
model.save("./Models/gesture_model.h5")
np.save("./Models/label_encoder.npy", encoder.classes_)

print("✅ Model training complete!")

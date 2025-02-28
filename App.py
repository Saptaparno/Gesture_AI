from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as tflite

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="models/gesture_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("landmarks")  # Expecting hand landmarks as JSON
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Preprocess input
    input_data = np.array(data, dtype=np.float32).reshape(input_details[0]["shape"])

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = int(np.argmax(output_data))

    return jsonify({"gesture": predicted_label, "confidence": float(np.max(output_data))})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

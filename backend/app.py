from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load CNN model and labels
cnn_model = tf.keras.models.load_model('backend/model_cnn.h5')
with open('backend/labels.json', 'r') as f:
    labels = json.load(f)

# Load LSTM model
lstm_model = tf.keras.models.load_model('backend/lstm_model.h5')

# Load LSTM input data
X_lstm = np.load('backend/dataset/X.npy')
last_sequence = X_lstm[-1].reshape(1, 3, 1)

# --- LSTM Prediction ---
def predict_with_lstm():
    pred_probs = lstm_model.predict(last_sequence)[0]
    predicted_index = np.argmax(pred_probs)
    confidence = float(pred_probs[predicted_index])
    return predicted_index, confidence

# --- CNN Prediction for single image ---
def predict_with_cnn(image):
    image = image.resize((128, 128)).convert('RGB')
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 128, 128, 3)

    pred_probs = cnn_model.predict(image_array)[0]
    predicted_index = np.argmax(pred_probs)
    confidence = float(pred_probs[predicted_index])

    # Find label name from predicted index
    predicted_label = [k for k, v in labels.items() if v == predicted_index][0]
    return predicted_label, confidence

# --- Combine CNN + LSTM ---
def combine_predictions(cnn_labels, cnn_confs, lstm_number, lstm_conf):
    # Take the most frequent CNN label
    if cnn_labels:
        cnn_label_final = max(set(cnn_labels), key=cnn_labels.count)
        cnn_conf_avg = round(sum(cnn_confs) / len(cnn_confs), 2)
    else:
        cnn_label_final = "Unknown"
        cnn_conf_avg = 0.0

    final_confidence = round((cnn_conf_avg + lstm_conf) / 2, 2)
    safe = cnn_conf_avg > 0.6 and lstm_conf > 0.6

    return {
        "number": lstm_number,
        "color_size": cnn_label_final,
        "confidence": final_confidence,
        "safe": safe
    }

# --- Prediction endpoint ---
@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    return predict()
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    cnn_labels = []
    cnn_confs = []

    for file in files:
        try:
            image = Image.open(io.BytesIO(file.read()))
            label, conf = predict_with_cnn(image)
            cnn_labels.append(label)
            cnn_confs.append(conf)
        except Exception as e:
            print("❌ Error processing image:", str(e))  # 👈 Console log
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500


    lstm_number, lstm_conf = predict_with_lstm()
    result = combine_predictions(cnn_labels, cnn_confs, lstm_number, lstm_conf)

    return jsonify(result)

# --- Run app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))


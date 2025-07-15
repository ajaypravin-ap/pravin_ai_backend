import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Dummy model for testing before training
def fake_model_predict(x):
    return [[0.72, 0.63, 0.84, 0.24, 0.88]]  # dummy prediction

model = type("FakeModel", (), {"predict": staticmethod(fake_model_predict)})

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("L").resize((100, 100))
        arr = np.array(img) / 255.0
        return arr.reshape(1, 100, 100, 1)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def predict_from_image(image_path):
    x = preprocess_image(image_path)
    if x is None:
        return None
    prediction = model.predict(x)[0]
    num1 = int(prediction[0] * 10)
    num2 = int(prediction[1] * 10)
    color = "Red" if prediction[2] > 0.5 else "Green"
    size = "Big" if prediction[3] > 0.5 else "Small"
    accuracy = round(prediction[4] * 100, 2)
    safety = "Don't Bet" if accuracy < 60 else "Safe"

    return {
        "number1": num1,
        "number2": num2,
        "color": color,
        "size": size,
        "accuracy": accuracy,
        "advice": safety
    }

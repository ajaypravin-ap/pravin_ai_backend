# predict.py

import joblib
from preprocess import extract_numbers_and_features
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Load and process test image
image_path = "../test_images/bht_chart.jpg"  # Use your image here
print(f"🖼️ Loading and processing: {image_path}")
data = extract_numbers_and_features(image_path)

# Only use numbers for prediction
# Encode 'size' and 'color'
size_map = {'Small': 0, 'Big': 1}
color_map = {'Red': 0, 'Green': 1}

# Convert each entry into [number, size, color] as numbers
X_test = np.array([
    [d['number'], size_map[d['size']], color_map[d['color']]]
    for d in data
])


# Predict
predictions = model.predict(X_test)

# Show top 2 most frequent predictions
from collections import Counter
counter = Counter(predictions)
most_common = counter.most_common(2)

print("\n🔮 Predicted Numbers:")
for number, count in most_common:
    size = "Big" if number >= 5 else "Small"
    color = "Red" if number % 2 == 0 else "Green"
    confidence = round((count / len(predictions)) * 100, 2)

    print(f"Number: {number} → {size}, {color} | 🎯 Confidence: {confidence}%")

    if confidence < 25:
        print("⚠️ Advice: Don’t Bet (Low confidence)")
    else:
        print("✅ Advice: Safe to Bet")

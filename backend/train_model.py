# File: backend/train_model.py

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Dummy dataset: Replace this with real features/labels from your chart image processing
X = np.array([
    [1, 0, 3],
    [4, 2, 1],
    [3, 5, 2],
    [6, 1, 4],
    [2, 3, 5],
])
y = [0, 1, 0, 1, 1]  # 0 = Small, 1 = Big (example classification)

# Split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to backend/model.pkl
with open('backend/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to backend/model.pkl")

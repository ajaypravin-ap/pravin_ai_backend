import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# ✅ Load preprocessed data
X = np.load('backend/dataset/X.npy')
y = np.load('backend/dataset/y.npy')

# ✅ Label encode y to ensure labels are 0 to N-1
le = LabelEncoder()
y = le.fit_transform(y)

# ✅ Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ One-hot encode target labels
num_classes = len(np.unique(y))
print("🧠 Number of classes:", num_classes)

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

# ✅ Define the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),  # e.g., (3,1)
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# ✅ Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    batch_size=16
)

# ✅ Save model
model.save('backend/lstm_model.h5')
print("✅ LSTM model saved to backend/lstm_model.h5")

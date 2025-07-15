import pandas as pd
import numpy as np
import os

# ✅ Correct path
input_csv_path = 'backend/dataset/chart_data.csv'
output_dir = 'backend/dataset'

# ✅ Check if file exists
if not os.path.exists(input_csv_path):
    print(f"❌ CSV not found at {input_csv_path}")
    exit()

df = pd.read_csv(input_csv_path)

# ✅ Check if required columns exist
required_columns = {'val1', 'val2', 'val3', 'label'}
if not required_columns.issubset(df.columns):
    print(f"❌ CSV missing required columns: {required_columns}")
    exit()

# ✅ Extract features and labels
X = df[['val1', 'val2', 'val3']].values
y = df['label'].values

# ✅ Reshape for LSTM input (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ✅ Save the data
np.save(os.path.join(output_dir, 'X.npy'), X)
np.save(os.path.join(output_dir, 'y.npy'), y)

print(f"✅ Saved {len(X)} samples to X.npy and y.npy")
print("Shape of X:", X.shape, ", Shape of y:", y.shape)

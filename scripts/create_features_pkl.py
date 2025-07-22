import numpy as np
import pickle
import os

input_path = "data/processed/flickr8k_features.npz"
output_path = "data/processed/sample_features/features.pkl"

# Make sure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

features = np.load(input_path)

# Convert keys to match the model runner expectation (no .jpg)
sample = {}
for k in list(features.keys())[:10]:
    clean_key = os.path.splitext(k)[0]  # removes .jpg
    sample[clean_key] = features[k]

# Save cleaned keys to pickle
with open(output_path, "wb") as f:
    pickle.dump(sample, f)

print("✅ Saved 10 cleaned image features to features.pkl")

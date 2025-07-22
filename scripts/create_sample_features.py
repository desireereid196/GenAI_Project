# scripts/create_sample_features.py

import numpy as np
import os

input_path = "data/processed/flickr8k_features.npz"
output_dir = "data/processed/sample_features"
os.makedirs(output_dir, exist_ok=True)

features = np.load(input_path)

# Save first 10 entries as .npy files
for idx, key in enumerate(features.files[:10]):
    np.save(os.path.join(output_dir, key.replace(".jpg", ".npy")), features[key])

print("✅ Sample features saved to data/processed/sample_features/")

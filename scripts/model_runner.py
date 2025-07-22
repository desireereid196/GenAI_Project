# scripts/model_runner.py

import os
import pickle
import numpy as np
from vtt.models.predict import generate_caption_greedy
from vtt.loaders import load_model_and_tokenizer
from vtt.utils import detect_and_set_device

# === Detect device ===
device = detect_and_set_device()
print(f"📟 Using device: {device}")

# === Load model and tokenizer ===
model, tokenizer = load_model_and_tokenizer()

# === Load image features from pickle ===
features_path = "data/processed/sample_features/features.pkl"
if not os.path.exists(features_path):
    raise FileNotFoundError(f"❌ Missing file: {features_path}")

with open(features_path, "rb") as f:
    features = pickle.load(f)

# === Generate captions ===
captions = []
print("\n🧠 Generating captions for sample images...\n")
for img_name in list(features.keys()):
    feature_vec = features[img_name]
    caption = generate_caption_greedy(model, tokenizer, feature_vec)
    print(f"🖼️ {img_name} → {caption}")
    captions.append(f"{img_name} --> {caption}")

# === Save to outputs ===
os.makedirs("outputs", exist_ok=True)
output_file = "outputs/sample_predictions.txt"
with open(output_file, "w") as f:
    f.write("\n".join(captions))

print(f"\n✅ Done! Captions saved to {output_file}")

# scripts/train_model.py

import os
import numpy as np
import pickle

from tensorflow.keras.callbacks import ModelCheckpoint
from vtt.models.decoder import build_decoder_model

# === Load image feature dictionary ===
feature_path = "data/processed/flickr8k_features.npz"
feature_data = np.load(feature_path)

# === Load image filenames in order ===
with open("data/processed/train_images.txt", "r") as f:
    image_ids = [line.strip() for line in f.readlines()]

# === Load Features ===
features = np.load("data/processed/flickr8k_features.npz")


# === Load Captions ===
captions = np.load("data/processed/flickr8k_padded_caption_sequences.npz")


# Sort to ensure matching order
image_keys = sorted(features.files)
caption_keys = sorted(captions.files)

X_train_img = []
y_train = []

for key in image_keys:
    if key in captions:
        for caption_seq in captions[key]:
            X_train_img.append(features[key])
            y_train.append(caption_seq)

X_train_img = np.array(X_train_img)
y_train = np.array(y_train)


# === Load tokenizer ===
with open("data/processed/flickr8k_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_len = y_train.shape[1] - 1


# === Build model ===
model = build_decoder_model(vocab_size=vocab_size, max_caption_len=max_len)

# === Save best model only ===
os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint(
    filepath="models/best_model.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)

# === Train model ===
model.fit(
    [X_train_img, y_train[:, :-1]],
    y_train[:, 1:],
    epochs=20,
    batch_size=64,
    callbacks=[checkpoint],
)

print("✅ Training complete. Best model saved to models/best_model.h5")

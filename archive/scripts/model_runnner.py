import os
import random

import pandas as pd

# Paths
caption_path = "../data/raw/flickr8k_captions.csv"
output_dir = "../data/processed/"
train_file = os.path.join(output_dir, "train_images.txt")
val_file = os.path.join(output_dir, "val_images.txt")

# Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(caption_path)

# Group captions by image
image_caption_map = df.groupby("image")["caption"].apply(list).to_dict()

# Shuffle and split
image_list = list(image_caption_map.keys())
random.shuffle(image_list)

split_idx = int(0.8 * len(image_list))
train_images = image_list[:split_idx]
val_images = image_list[split_idx:]

# Save splits
with open(train_file, "w") as f:
    f.write("\n".join(train_images))

with open(val_file, "w") as f:
    f.write("\n".join(val_images))

print(f"✅ CSV Split complete: {len(train_images)} train, {len(val_images)} val")

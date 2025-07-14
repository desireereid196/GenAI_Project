"""
config.py

Global configuration constants used across the project.

This module centralizes parameters and special tokens to ensure consistency 
across different components including preprocessing, tokenization, and model input.

Constants:
    - Special Tokens (e.g., <start>, <end>, <unk>)
    - ImageNet normalization stats for ResNet preprocessing
    - Vocabulary frequency threshold
    - Target image input size
"""

# ------------------------
# Special Tokens
# ------------------------

# Token used to denote the start of a caption sequence
START_TOKEN = "startseq"

# Token used to denote the end of a caption sequence
END_TOKEN = "endseq"

# Token representing out-of-vocabulary words during training or inference
OOV_TOKEN = "<unk>"

# ------------------------
# Image Preprocessing (for ResNet50)
# ------------------------

# ImageNet dataset mean values for each RGB channel (used to standardize pixel values)
IMAGENET_MEAN = [0.485, 0.456, 0.406]

# ImageNet dataset standard deviation values for each RGB channel
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target input size (height, width) for ResNet50 model
IMAGE_SIZE = (224, 224)

# ------------------------
# Vocabulary Thresholding
# ------------------------

# Minimum word frequency required to include a word in the vocabulary
MIN_WORD_FREQ = 5

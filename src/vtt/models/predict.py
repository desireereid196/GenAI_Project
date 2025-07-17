"""
predict.py

This module provides functions for generating image captions using a trained
image captioning model and displaying the results alongside their corresponding images.

Key functionality includes:
- Greedy decoding: Generate captions one token at a time by selecting the most probable
  word.
- Visualization: Display images with predicted captions using matplotlib.

Usage example:
    caption = generate_caption_greedy(model, tokenizer, features["image123.jpg"])
    display_images_with_captions(["image123.jpg"], model, tokenizer, features,
        "data/flickr8k_images_dataset/Images")
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def generate_caption_greedy(
    model: Model,
    tokenizer: Tokenizer,
    image_feature: np.ndarray,
    max_len: Optional[int] = None,
) -> str:
    """
    Generates a caption from image features using greedy decoding.

    Args:
        model (Model): Trained image captioning model.
        tokenizer (Tokenizer): Tokenizer used to encode/decode tokens.
        image_feature (np.ndarray): Precomputed image feature vector (shape: (2048,)).
        max_len (Optional[int]): Maximum caption length. If None, inferred from model
        input.

    Returns:
        str: Generated caption (excluding <startseq> and <endseq> tokens).
    """
    # Infer max caption length from model input shape
    if max_len is None:
        max_len = model.input[1].shape[1]  # From caption_input layer

    # Get special tokens
    start_token = tokenizer.word_index.get("startseq")
    end_token = tokenizer.word_index.get("endseq")

    if start_token is None or end_token is None:
        raise ValueError("Tokenizer must contain <startseq> and <endseq> tokens.")

    # Start generating from <startseq>
    caption_seq = [start_token]

    for _ in range(max_len):
        # Pad input sequence
        padded_seq = pad_sequences([caption_seq], maxlen=max_len, padding="post")

        # Predict next word using current sequence
        preds = model.predict(
            [np.expand_dims(image_feature, axis=0), padded_seq], verbose=0
        )

        # Take most probable word ID at the current time step
        next_id = int(np.argmax(preds[0, len(caption_seq) - 1, :]))

        # Append to sequence and break if <endseq> predicted
        caption_seq.append(next_id)
        if next_id == end_token:
            break

    # Decode token IDs to words, skipping start/end tokens
    index_word = tokenizer.index_word
    words = [index_word.get(i, "") for i in caption_seq[1:] if i != end_token]

    return " ".join(words).strip()


def display_images_with_captions(
    image_ids: List[str],
    model: Model,
    tokenizer: Tokenizer,
    features: Dict[str, np.ndarray],
    image_folder: str,
) -> None:
    """
    Displays each image with its predicted caption.

    Args:
        image_ids (List[str]): List of image filenames to visualize.
        model (Model): Trained captioning model.
        tokenizer (Tokenizer): Tokenizer used for caption generation.
        features (Dict[str, np.ndarray]): Dictionary of precomputed image features
            keyed by image ID.
        image_folder (str): Path to the folder containing the original images.
    """
    for image_id in image_ids:
        try:
            # Get image feature and generate caption
            image_feature = features[image_id]
            caption = generate_caption_greedy(model, tokenizer, image_feature)

            # Load and display the image
            image_path = f"{image_folder}/{image_id}"
            img = Image.open(image_path)

            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(caption, fontsize=12)
            plt.show()

        except Exception as e:
            print(f"Error displaying {image_id}: {e}")

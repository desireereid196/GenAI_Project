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
from PIL import Image
from typing import List, Dict
import os
import textwrap
import math
import logging


# Configure module-specific logger
logger = logging.getLogger(__name__)


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


def generate_caption_beam(model, tokenizer, image_feature, max_len, beam_width=3):
    """Generate a caption using beam search.

    Args:
        model: Trained image captioning model.
        tokenizer: Tokenizer used to encode/decode tokens.
        image_feature (np.ndarray): Precomputed image feature vector.
        max_len (int): Maximum length of the generated caption.
        beam_width (int, optional): Beam width for beam search. Defaults to 3.

    Returns:
        str: Generated caption (excluding <startseq> and <endseq> tokens).
    """
    start_seq = [tokenizer.word_index["startseq"]]
    sequences = [(start_seq, 0.0)]

    end_token = tokenizer.word_index["endseq"]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            # stop expanding if last token is <endseq>
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            pad_seq = pad_sequences([seq], maxlen=max_len, padding="post")
            preds = model.predict([image_feature[None, :], pad_seq], verbose=0)[0]
            preds = preds[len(seq) - 1]  # current timestep

            # take top beam_width predictions
            top_ids = np.argsort(preds)[-beam_width:]
            for wid in top_ids:
                candidate = (seq + [wid], score + np.log(preds[wid] + 1e-10))
                all_candidates.append(candidate)

        # keep beam_width best
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[
            :beam_width
        ]

    # return the sequence with highest score
    best_seq = sequences[0][0]

    # convert to words, drop <startseq>/<endseq>
    words = [
        tokenizer.index_word.get(i, "")
        for i in best_seq
        if i not in (start_seq[0], end_token, 0)
    ]
    return " ".join(words).strip()


def _plot_images_with_captions(
    image_ids: List[str],
    captions: List[str],
    image_folder: str,
    cols: int = 3,
    title: str = "Generated Captions",
) -> None:
    """
    Plot images and captions in a grid layout, handling spacing and wrapping.

    Args:
        image_ids (List[str]): List of image filenames.
        captions (List[str]): Corresponding captions for each image.
        image_folder (str): Path to the folder containing the images.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
        title (str, optional): Overall title for the plot. Defaults to "Generated Captions".

    Returns:
        None
    """
    rows = math.ceil(len(image_ids) / cols)
    plt.figure(figsize=(cols * 5, rows * 5))

    for i, (image_id, caption) in enumerate(zip(image_ids, captions), 1):
        try:
            img_path = os.path.join(image_folder, image_id)
            img = Image.open(img_path)

            ax = plt.subplot(rows, cols, i)
            ax.imshow(img)
            ax.axis("off")

            # Wrap caption to avoid overlap
            wrapped_caption = "\n".join(textwrap.wrap(caption, width=40))
            ax.set_title(wrapped_caption, fontsize=10)

        except Exception as e:
            print(f"Error displaying {image_id}: {e}")

    plt.suptitle(title, fontsize=16)

    # Adjust spacing to avoid title/caption overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4)
    plt.show()


def display_images_with_greedy_captions(
    image_ids: List[str],
    model,
    tokenizer,
    features: Dict[str, np.ndarray],
    image_folder: str,
    cols: int = 3,
) -> None:
    """
    Wrapper to generate and display image captions using greedy decoding.

    """

    captions = []
    for image_id in image_ids:
        feature = features[image_id]
        caption = generate_caption_greedy(model, tokenizer, feature)
        caption = caption.replace("startseq", "").replace("endseq", "").strip()
        captions.append(caption)

    _plot_images_with_captions(
        image_ids=image_ids,
        captions=captions,
        image_folder=image_folder,
        cols=cols,
        title="Greedy Captions",
    )


def display_images_with_beam_captions(
    image_ids: List[str],
    model,
    tokenizer,
    features: Dict[str, np.ndarray],
    image_folder: str,
    beam_width: int = 5,
    max_len: int = 20,
    cols: int = 3,
) -> None:
    """
    Wrapper to generate and display image captions using beam search decoding.
    """
    captions = []
    for image_id in image_ids:
        feature = features[image_id]
        caption = generate_caption_beam(model, tokenizer, feature, max_len, beam_width)
        caption = caption.replace("startseq", "").replace("endseq", "").strip()
        captions.append(caption)

    _plot_images_with_captions(
        image_ids=image_ids,
        captions=captions,
        image_folder=image_folder,
        cols=cols,
        title=f"Beam Search Captions (beam={beam_width})",
    )

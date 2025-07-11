"""
data_loader.py

This module provides utilities for loading image features and
corresponding caption sequences, and preparing them as TensorFlow
`tf.data.Dataset` objects suitable for model training.

It handles loading data from `.npz` files, aligning features with captions
based on image IDs, and setting up batching and shuffling for efficient
data pipeline processing.
"""

import numpy as np
import tensorflow as tf

def load_features_and_sequences(
    features_path: str,
    captions_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000
) -> tf.data.Dataset:
    """Loads image features and corresponding caption sequences into a tf.data.Dataset.

    This function reads pre-extracted image features (e.g., from a ResNet) and
    padded caption sequences from `.npz` files. It aligns the features with
    their respective captions based on image IDs, then converts them into
    TensorFlow tensors to create a `tf.data.Dataset`. The dataset can be
    shuffled, batched, and prefetched for optimized training performance.

    Args:
        features_path (str): Path to the `.npz` file containing image features.
                             Expected format: {image_id: feature_vector_ndarray}.
        captions_path (str): Path to the `.npz` file containing caption sequences.
                             Expected format: {image_id: list_of_padded_caption_sequences_ndarray}.
                             `allow_pickle=True` is used for loading potentially complex arrays.
        batch_size (int): The number of elements per batch in the output dataset. Defaults to 32.
        shuffle (bool): If True, shuffles the dataset. Defaults to True.
        buffer_size (int): The number of elements from this dataset from which the new dataset
                           will sample. Only used if `shuffle` is True. A larger buffer_size
                           provides better shuffling but uses more CPU memory. Defaults to 1000.

    Returns:
        tf.data.Dataset: A TensorFlow dataset where each element is a tuple
                         (image_feature_tensor, caption_sequence_tensor).
                         The dataset is batched and prefetched.
    """
    # Load image features and caption sequences from .npz files
    features_npz = np.load(features_path)
    captions_npz = np.load(captions_path, allow_pickle=True)

    image_features = []
    caption_sequences = []

    # Iterate through image IDs in captions_npz to ensure captions have corresponding features
    for img_id in captions_npz.files:
        if img_id in features_npz:
            # For each image ID, append its feature for every associated caption
            for caption in captions_npz[img_id]:
                image_features.append(features_npz[img_id])
                caption_sequences.append(caption)

    # Convert the lists of features and captions to TensorFlow tensors
    features_tensor = tf.convert_to_tensor(image_features, dtype=tf.float32)
    captions_tensor = tf.convert_to_tensor(caption_sequences, dtype=tf.int32)

    # Create a TensorFlow dataset from the tensors
    dataset = tf.data.Dataset.from_tensor_slices((features_tensor, captions_tensor))

    # Apply shuffling if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch the dataset and prefetch elements for performance
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def prepare_training_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Prepares a dataset for sequence-to-sequence training by splitting captions into inputs and targets.

    This function applies a mapping operation to each element of the input
    dataset. For each (image_feature, caption_sequence) pair, it transforms
    them into ((image_feature, input_caption_sequence), target_caption_sequence).
    The input caption sequence is the original caption shifted by one position
    (excluding the last token), and the target caption sequence is the original
    caption shifted by one position (excluding the first token). This is
    standard for teacher-forcing in sequence models where the model learns
    to predict the next token.

    Args:
        dataset (tf.data.Dataset): The input TensorFlow dataset, where each
                                   element is a tuple (image_feature_tensor,
                                   caption_sequence_tensor).

    Returns:
        tf.data.Dataset: A new TensorFlow dataset where each element is a tuple
                         of ((image_feature_tensor, input_caption_tensor), target_caption_tensor).
    """
    def split_inputs_and_targets(img, caption):
        """Helper function to split a caption into input and target sequences."""
        # Input sequence: all tokens except the last one (e.g., <start> word1 word2 ...)
        # Target sequence: all tokens except the first one (e.g., word1 word2 <end>)
        return (img, caption[:-1]), caption[1:]
    return dataset.map(split_inputs_and_targets)
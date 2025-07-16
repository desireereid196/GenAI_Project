"""
data_loader.py

This module provides utilities for loading and preparing image-caption datasets
as TensorFlow `tf.data.Dataset` objects suitable for training, validation, and testing.

It handles loading features and caption sequences from `.npz` files, aligning features
with captions based on image IDs, splitting into train/validation/test, preparing the
data for sequence-to-sequence learning, and setting up batching and shuffling for
efficient data pipeline processing.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union


def load_features_and_sequences(
    features_path: str, captions_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load aligned image features and caption sequences from .npz files.

    Args:
        features_path (str): Path to the .npz file of image features {image_id: feature_vector}.
        captions_path (str): Path to the .npz file of padded caption sequences
                             {image_id: [sequence1, sequence2, ...]}.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two NumPy arrays (features, captions), aligned 1:1.
    """
    features_npz = np.load(features_path)
    captions_npz = np.load(captions_path, allow_pickle=True)

    image_features = []
    caption_sequences = []

    for img_id in captions_npz.files:
        if img_id in features_npz:
            for caption in captions_npz[img_id]:
                image_features.append(features_npz[img_id])
                caption_sequences.append(caption)

    return (
        np.asarray(image_features, dtype=np.float32),
        np.asarray(caption_sequences, dtype=np.int32),
    )


def prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Split each caption into (input, target) for sequence-to-sequence training.

    Args:
        dataset (tf.data.Dataset): A dataset of (image_feature, full_caption) pairs.

    Returns:
        tf.data.Dataset: A dataset of ((image_feature, input_caption), target_caption) pairs.
    """

    def split_inputs_and_targets(img, caption):
        return (img, caption[:-1]), caption[1:]

    return dataset.map(split_inputs_and_targets, num_parallel_calls=tf.data.AUTOTUNE)


def load_training_dataset(
    features_path: str,
    captions_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000,
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Loads and returns a fully-prepared tf.data.Dataset ready for training.
    It handles loading, aligning, splitting, batching, and prefetching.

    Args:
        features_path (str): Path to image features .npz file.
        captions_path (str): Path to padded caption sequences .npz file.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the dataset before training.
        buffer_size (int): Buffer size for shuffling.
        cache (bool): Whether to cache the dataset in memory. Defaults to False.

    Returns:
        tf.data.Dataset: Batches of ((image, caption_input), caption_target)
    """
    # Load raw aligned feature-caption pairs
    image_features, caption_sequences = load_features_and_sequences(
        features_path, captions_path
    )
    assert len(image_features) == len(
        caption_sequences
    ), "Mismatched features and captions."

    # Split into inputs and targets
    dataset = tf.data.Dataset.from_tensor_slices((image_features, caption_sequences))
    dataset = prepare_dataset(dataset)

    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    # Apply batching and prefetching in correct order
    return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def load_split_datasets(
    features_path: str,
    captions_path: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    shuffle: bool = True,
    buffer_size: int = 1000,
    seed: Optional[int] = 42,
    cache: bool = False,
    return_numpy: bool = False,
) -> Union[
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Load features and captions, perform a train/val/test split, and return preprocessed tf.data.Datasets
    or optionally the raw NumPy arrays.

    Args:
        features_path (str): Path to the .npz file with image features.
        captions_path (str): Path to the .npz file with padded caption sequences.
        batch_size (int): Number of samples per batch. Defaults to 32.
        val_split (float): Proportion of data to use for validation. Defaults 0.15.
        test_split (float): Proportion of data to use for testing. Defaults to 0.10.
        shuffle (bool): Whether to shuffle the data before splitting. Defaults to True
        buffer_size (int): Buffer size for shuffling. Defaults to 1000.
        seed (int, optional): Random seed for shuffling. Defaults to 42.
        cache (bool): Whether to cache the dataset in memory. Defaults to False.
        return_numpy (bool): If True, return NumPy arrays instead of tf.data.Dataset objects. Defaults to False.

    Returns:
        If return_numpy is False:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
        If return_numpy is True:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (train_X, train_y, val_X, val_y, test_X, test_y)
    """
    image_features, caption_sequences = load_features_and_sequences(
        features_path, captions_path
    )
    assert len(image_features) == len(
        caption_sequences
    ), "Mismatched features and captions."

    # Shuffle before splitting
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(image_features))
        image_features = image_features[idx]
        caption_sequences = caption_sequences[idx]

    # Compute split boundaries
    total = len(image_features)
    val_size = int(val_split * total)
    test_size = int(test_split * total)
    train_size = total - val_size - test_size

    # Split datasets
    train_X = image_features[:train_size]
    train_y = caption_sequences[:train_size]
    val_X = image_features[train_size : train_size + val_size]
    val_y = caption_sequences[train_size : train_size + val_size]
    test_X = image_features[train_size + val_size :]
    test_y = caption_sequences[train_size + val_size :]

    if return_numpy:
        return train_X, train_y, val_X, val_y, test_X, test_y

    def make_dataset(
        features: np.ndarray, captions: np.ndarray, drop_remainder: bool
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset pipeline from NumPy arrays, optionally caching,
        shuffling, batching, and prefetching the data.

        Args:
            features (np.ndarray): Image feature vectors.
            captions (np.ndarray): Corresponding padded caption sequences.
            drop_remainder (bool): Whether to drop the last batch if it is smaller than `batch_size`.

        Returns:
            tf.data.Dataset: A batched, prefetched dataset suitable for training or evaluation.
        """
        dataset = tf.data.Dataset.from_tensor_slices((features, captions))
        dataset = prepare_dataset(dataset)

        if cache:
            dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        return dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(
            tf.data.AUTOTUNE
        )

    return (
        make_dataset(
            train_X, train_y, True
        ),  # drop_remainder = True for training; needed to ensure consistent batch shape
        make_dataset(val_X, val_y, False),  # keep all data for validation
        make_dataset(test_X, test_y, False),  # keep all data for testing
    )

"""
data_loader.py

This module provides essential functionalities for loading and preparing
image features and corresponding caption sequences for image captioning tasks.
It handles loading data from preprocessed .npz files, aligning features with
captions, splitting the dataset into training, validation, and test sets,
and converting data into optimized TensorFlow `tf.data.Dataset` objects
or NumPy arrays.

Key functionalities include:
    - Loading aligned image features and caption sequences from disk.
    - Preparing dataset elements for sequence-to-sequence training (input/target splitting).
    - Splitting the full dataset into train, validation, and test sets with options
      for shuffling, caching, and returning data as TensorFlow Datasets or NumPy arrays.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union, List
import logging

# Configure module-specific logger
logger = logging.getLogger(__name__)


def load_features_and_sequences(
    features_path: str, captions_path: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load aligned image features and caption sequences from .npz files,
    along with associated image IDs for each (feature, caption) pair.

    This function iterates through the captions, and for each caption, it
    retrieves the corresponding image feature. This means if an image has
    multiple captions, its feature vector and image ID will be duplicated
    for each of its captions in the output arrays/list.

    Args:
        features_path (str): Path to the .npz file containing pre-extracted
                             image features (e.g., from a CNN encoder).
                             Expected format: {image_id: feature_vector_ndarray}.
        captions_path (str): Path to the .npz file containing preprocessed
                             caption sequences (e.g., padded token IDs).
                             Expected format: {image_id: list_of_padded_caption_sequences_ndarray}.
                             `allow_pickle=True` is often used when saving/loading these.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: A tuple containing:
            - image_features (np.ndarray): A 2D NumPy array where each row is an image
                                           feature vector. Duplicate features will exist
                                           for images with multiple captions.
            - caption_sequences (np.ndarray): A 2D NumPy array where each row is a padded
                                              caption sequence (token IDs).
            - image_ids (List[str]): A list of image IDs, where each ID corresponds to
                                     the image feature and caption sequence at the same index.
                                     Duplicate IDs will exist for images with multiple captions.
    """
    logger.info(f"Loading image features from: {features_path}")
    features_npz = np.load(features_path)
    logger.info(f"Loading caption sequences from: {captions_path}")
    captions_npz = np.load(captions_path, allow_pickle=True)

    image_features = []
    caption_sequences = []
    image_ids = []

    # Iterate through image IDs that have captions
    for img_id in captions_npz.files:
        # Ensure the image also has corresponding features
        if img_id in features_npz:
            # For each caption associated with the current image ID
            for caption in captions_npz[img_id]:
                # Append the image feature and caption sequence
                image_features.append(features_npz[img_id])
                caption_sequences.append(caption)
                image_ids.append(
                    img_id
                )  # Store the image ID for this specific (feature, caption) pair
        else:
            logger.warning(
                f"Image ID '{img_id}' found in captions but not in features. Skipping."
            )

    # Convert lists to NumPy arrays for efficient processing
    # Ensure float32 for features (common for neural networks) and int32 for token IDs.
    return (
        np.asarray(image_features, dtype=np.float32),
        np.asarray(caption_sequences, dtype=np.int32),
        image_ids,
    )


def prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Prepares a TensorFlow dataset for sequence-to-sequence training.

    This function transforms each element `(image_feature, caption_sequence, image_id)`
    into `((image_feature, caption_input), caption_target, image_id)`.
    The `caption_input` is the original caption sequence shifted by one position
    (excluding the last token), and the `caption_target` is the original caption
    sequence shifted by one position (excluding the first token). This is standard
    for teacher-forcing in sequence models where the model learns to predict the next token.
    The `image_id` is passed through for potential use in evaluation.

    Args:
        dataset (tf.data.Dataset): The input TensorFlow dataset, where each element
                                   is a tuple `(image_feature_tensor, caption_sequence_tensor, image_id_tensor)`.

    Returns:
        tf.data.Dataset: A new TensorFlow dataset where each element is a tuple
                         of `((image_feature_tensor, caption_input_tensor), caption_target_tensor, image_id_tensor)`.
    """

    def split_inputs_and_targets(
        img_feature: tf.Tensor, caption: tf.Tensor, img_id: tf.Tensor
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Helper function to split a caption into input and target sequences for training.
        """
        # caption[:-1] serves as the input sequence (e.g., <start> word1 word2 ...)
        # caption[1:] serves as the target sequence (e.g., word1 word2 <end>)
        return (img_feature, caption[:-1]), caption[1:], img_id

    # Apply the transformation to each element in the dataset in parallel
    return dataset.map(split_inputs_and_targets, num_parallel_calls=tf.data.AUTOTUNE)


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
    Tuple[
        Tuple[
            np.ndarray, np.ndarray, List[str]
        ],  # train_data: (features, captions, ids)
        Tuple[np.ndarray, np.ndarray, List[str]],  # val_data: (features, captions, ids)
        Tuple[
            np.ndarray, np.ndarray, List[str]
        ],  # test_data: (features, captions, ids)
    ],
]:
    """
    Load preprocessed image features and caption sequences, and split them
    into training, validation, and test sets.

    The data can be returned either as optimized TensorFlow `tf.data.Dataset` objects
    (suitable for direct model training) or as raw NumPy arrays and Python lists.

    Args:
        features_path (str): Path to the .npz file containing image features.
        captions_path (str): Path to the .npz file containing padded caption sequences.
        batch_size (int): The number of elements per batch in the output TensorFlow datasets.
                          Defaults to 32. Only used if `return_numpy` is False.
        val_split (float): The proportion of the dataset to use for the validation set.
                           Defaults to 0.15 (15%).
        test_split (float): The proportion of the dataset to use for the test set.
                            Defaults to 0.10 (10%).
        shuffle (bool): If True, shuffles the dataset before splitting. Defaults to True.
        buffer_size (int): The number of elements from the dataset to buffer for shuffling.
                           Only relevant if `shuffle` is True and `return_numpy` is False.
                           A larger buffer provides better shuffling but uses more memory.
                           Defaults to 1000.
        seed (Optional[int]): Random seed for reproducibility of shuffling. Defaults to 42.
        cache (bool): If True, caches the TensorFlow datasets in memory after the first epoch
                      for faster subsequent epochs. Only used if `return_numpy` is False.
                      Consider `dataset.take(k).cache().repeat()` for specific caching needs.
        return_numpy (bool): If True, returns the splits as raw NumPy arrays and lists.
                             If False, returns them as batched and prefetched `tf.data.Dataset` objects.

    Returns:
        Union[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
              Tuple[Tuple[np.ndarray, np.ndarray, List[str]], ...]]:
            A tuple containing the train, validation, and test data.
            - If `return_numpy` is True: Each split is a tuple of (features_array, captions_array, image_ids_list).
            - If `return_numpy` is False: Each split is a `tf.data.Dataset` object.
    """
    logger.info("Loading features and sequences for dataset splitting...")
    image_features, caption_sequences, image_ids = load_features_and_sequences(
        features_path, captions_path
    )
    # Assert that all lists/arrays have the same length, ensuring alignment
    assert (
        len(image_features) == len(caption_sequences) == len(image_ids)
    ), "Mismatch in lengths of image features, caption sequences, or image IDs."

    if shuffle:
        logger.info(f"Shuffling data with seed: {seed}")
        rng = np.random.default_rng(seed)  # Use modern NumPy random generator
        idx = rng.permutation(
            len(image_features)
        )  # Get a random permutation of indices

        # Apply the same permutation to all three arrays/lists to maintain alignment
        image_features = image_features[idx]
        caption_sequences = caption_sequences[idx]
        image_ids = [image_ids[i] for i in idx]  # Apply permutation to list of IDs

    total_samples = len(image_features)
    val_size = int(val_split * total_samples)
    test_size = int(test_split * total_samples)
    # The remaining samples go to the training set
    train_size = total_samples - val_size - test_size

    logger.info("\n--- Dataset Split Sizes (number of individual samples) ---")
    logger.info(f"Total samples loaded: {total_samples}")
    logger.info(f"Train samples: {train_size}")
    logger.info(f"Validation samples: {val_size}")
    logger.info(f"Test samples: {test_size}")
    logger.info("----------------------------------------------------------\n")

    # Split the data into train, validation, and test sets based on calculated sizes
    train_X, train_y, train_ids = (
        image_features[:train_size],
        caption_sequences[:train_size],
        image_ids[:train_size],
    )
    val_X, val_y, val_ids = (
        image_features[train_size : train_size + val_size],
        caption_sequences[train_size : train_size + val_size],
        image_ids[train_size : train_size + val_size],
    )
    test_X, test_y, test_ids = (
        image_features[train_size + val_size :],
        caption_sequences[train_size + val_size :],
        image_ids[train_size + val_size :],
    )

    if return_numpy:
        logger.info("Returning dataset splits as NumPy arrays and lists.")
        return (
            (train_X, train_y, train_ids),
            (val_X, val_y, val_ids),
            (test_X, test_y, test_ids),
        )

    # If not returning NumPy, create TensorFlow tf.data.Dataset objects
    logger.info("Creating TensorFlow tf.data.Dataset objects for splits.")

    def make_dataset(
        features: np.ndarray, captions: np.ndarray, ids: List[str], drop_remainder: bool
    ) -> tf.data.Dataset:
        """Helper function to create and configure a tf.data.Dataset."""
        # Create a dataset from slices of NumPy arrays and list
        ds = tf.data.Dataset.from_tensor_slices((features, captions, ids))

        # Prepare the dataset for seq2seq training (splitting captions into input/target)
        ds = prepare_dataset(ds)

        if cache:
            # Caching the dataset in memory can significantly speed up training
            # after the first epoch. Be mindful of memory usage for very large datasets.
            # If .repeat() is used later, consider .take().cache().repeat() to avoid warnings.
            ds = ds.cache()
            logger.debug("Dataset caching enabled.")
        if shuffle:
            # Shuffle the dataset. Buffer size determines the randomness quality.
            ds = ds.shuffle(buffer_size=buffer_size)
            logger.debug(f"Dataset shuffling enabled with buffer size: {buffer_size}.")

        # Batch the dataset and prefetch elements for optimal pipeline performance.
        # `drop_remainder=True` is often used for training to ensure all batches have
        # the same size, which can simplify model input shapes.
        return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(
            tf.data.AUTOTUNE
        )

    # Create and return the TensorFlow datasets for each split
    train_ds = make_dataset(train_X, train_y, train_ids, drop_remainder=True)
    val_ds = make_dataset(val_X, val_y, val_ids, drop_remainder=False)
    test_ds = make_dataset(test_X, test_y, test_ids, drop_remainder=False)

    logger.info("TensorFlow datasets created and configured.")
    return (train_ds, val_ds, test_ds)

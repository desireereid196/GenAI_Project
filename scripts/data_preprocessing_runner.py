"""
data_runner.py

This script orchestrates the entire data preparation pipeline for an image captioning project.
It handles image feature extraction, caption preprocessing (cleaning, tokenization, padding),
and splitting the processed data into training, validation, and test datasets.
"""

import numpy as np
import logging

from vtt.data.data_loader import load_split_datasets
from vtt.data.caption_preprocessing import (
    load_and_clean_captions,
    filter_captions_by_frequency,
    fit_tokenizer,
    captions_to_sequences,
    compute_max_caption_length,
    pad_caption_sequences,
    save_padded_sequences,
    save_tokenizer,
)
from vtt.data.image_preprocessing import (
    extract_features_from_directory,
    save_features,
    load_features,
)
from vtt.utils.config import END_TOKEN, OOV_TOKEN, START_TOKEN


# Configure logging for this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def pre_process_images(image_dir: str, output_features_path: str):
    """
    Pre-processes images by extracting features using a pre-trained CNN model
    and saving them to a specified output path.

    Args:
        image_dir (str): Path to the directory containing raw images.
        output_features_path (str): Path where the extracted image features
                                     (e.g., as a .npz file) will be saved.
    """
    print(f"--- Starting image preprocessing from: {image_dir} ---")
    # Extract features for all images in the directory
    features = extract_features_from_directory(image_dir)

    # Save the full dictionary of features to disk
    save_features(features, output_features_path)
    print(f"--- Image features saved to: {output_features_path} ---")


def pre_process_captions(
    captions_path: str, padded_caption_sequences_path: str, tokenizer_path: str
):
    """
    Pre-processes raw captions by cleaning, tokenizing, filtering, padding,
    and saving the processed sequences and the tokenizer.

    Args:
        captions_path (str): Path to the raw captions file (e.g., .csv).
        padded_caption_sequences_path (str): Path where the padded caption sequences
                                             (e.g., as a .npz file) will be saved.
        tokenizer_path (str): Path where the fitted tokenizer (e.g., as a .json file)
                              will be saved.
    """
    print(f"\n--- Starting caption preprocessing from: {captions_path} ---")

    # Step 1: Load and clean raw captions (e.g., remove punctuation, convert to lowercase)
    print("Loading and cleaning captions...")
    captions_dict = load_and_clean_captions(captions_path)

    # Step 2: Filter out rare words and build initial vocabulary
    # min_word_freq=5 means words appearing less than 5 times will be ignored or replaced with OOV_TOKEN
    print("Filtering captions by word frequency (min_word_freq=5)...")
    filtered_captions, vocab = filter_captions_by_frequency(
        captions_dict, min_word_freq=5
    )
    print(f"\tVocabulary size after filtering: {len(vocab)}")

    # Step 3: Fit tokenizer on filtered captions to assign unique IDs to words
    # num_words=10000 limits the tokenizer to the 10,000 most frequent words
    print("Fitting tokenizer on filtered captions (max 10,000 words)...")
    tokenizer = fit_tokenizer(filtered_captions, num_words=10000)

    # Step 4: Convert cleaned captions to sequences of token IDs
    print("Converting captions to sequences of token IDs...")
    seqs = captions_to_sequences(filtered_captions, tokenizer)

    # Step 5: Compute max length for padding using 95th percentile
    # This helps ensure most captions fit without excessive padding, and very long
    # outlier captions don't disproportionately affect max_length.
    print("Computing optimal max caption length (95th percentile)...")
    max_length = compute_max_caption_length(seqs, quantile=0.95)
    print(f"\tCalculated max caption length: {max_length}")

    # Step 6: Pad all sequences to uniform length
    print(f"Padding sequences to uniform length ({max_length})...")
    padded_seqs = pad_caption_sequences(seqs, max_length=max_length)

    # Step 7: Save processed data and tokenizer for later use
    print("Saving processed sequences and tokenizer...")
    save_padded_sequences(padded_seqs, padded_caption_sequences_path)
    save_tokenizer(tokenizer, tokenizer_path)
    print(
        f"--- Processed caption sequences saved to: {padded_caption_sequences_path} ---"
    )
    print(f"--- Tokenizer saved to: {tokenizer_path} ---")


def get_processed_data(features_path: str, captions_path: str):
    """
    Loads pre-processed image features and caption sequences, and splits them
    into TensorFlow tf.data.Dataset objects for training, validation, and testing.

    Args:
        features_path (str): Path to the .npz file containing pre-extracted image features.
        captions_path (str): Path to the .npz file containing pre-processed and padded caption sequences.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
            A tuple containing the train, validation, and test tf.data.Dataset objects.
    """
    print(f"\n--- Loading and splitting datasets ---")
    # Load and split datasets into train, validation, and test sets
    # The batch_size, split ratios, shuffle, and cache parameters are configured here.
    # The return_numpy=False ensures TensorFlow tf.data.Dataset objects are returned.
    train_ds, val_ds, test_ds, _ = (
        load_split_datasets(  # Note: `_` used to ignore total_test_samples if not needed here
            features_path=features_path,
            captions_path=captions_path,
            batch_size=64,  # Batch size for training
            val_split=0.15,  # 15% of data for validation
            test_split=0.10,  # 10% of data for testing
            shuffle=True,  # Shuffle data before splitting
            buffer_size=1000,  # Buffer size for shuffling
            seed=42,  # Random seed for reproducibility
            cache=True,  # Cache datasets in memory for faster epoch iterations
            return_numpy=False,  # Return tf.data.Dataset objects
        )
    )
    print(f"--- Datasets loaded and split successfully ---")
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Define dataset-specific paths
    dataset_name = "flickr8k"

    # Directory containing raw images (subset for demonstration or actual full dataset)
    image_dir = f"../data/flickr8k_images/subset/"

    # Path to the raw captions CSV file
    raw_captions_path = f"../data/raw/{dataset_name}_captions.csv"

    # Path to save the pre-processed and padded caption sequences
    padded_sequences_path = (
        f"../data/processed/{dataset_name}_padded_caption_sequences.npz"
    )

    # Path to save the extracted image features
    features_path = f"../data/processed/{dataset_name}_features.npz"

    # Path to load the pre-processed captions (same as padded_sequences_path for consistency)
    captions_path = f"../data/processed/{dataset_name}_padded_caption_sequences.npz"

    # Path to save the fitted tokenizer
    tokenizer_path = f"../data/processed/{dataset_name}_tokenizer.json"

    print("--- Starting Data Preparation Runner ---")

    # Execute image preprocessing
    pre_process_images(image_dir, features_path)

    # Execute caption preprocessing
    pre_process_captions(raw_captions_path, padded_sequences_path, tokenizer_path)

    # Example of how to get the processed datasets (train, val, test) if needed for immediate use
    # train_dataset, val_dataset, test_dataset = get_processed_data(features_path, captions_path)
    # print(f"Train Dataset: {train_dataset}")
    # print(f"Validation Dataset: {val_dataset}")
    # print(f"Test Dataset: {test_dataset}")

    print("\n--- Data Preparation Runner Finished ---")

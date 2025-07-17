"""
caption_preprocessing.py

This module provides utilities for preprocessing image captions for tasks like
image captioning. It includes functions for cleaning, tokenizing, filtering,
padding, and saving/loading captions in various formats.

"""

import os
import pickle
import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from vtt.utils.config import END_TOKEN, OOV_TOKEN, START_TOKEN


def clean_caption(text: str) -> str:
    """Cleans and normalizes a single caption string.

    Converts to lowercase, removes punctuation (except apostrophes), and
    adds START and END tokens.

    Args:
        text (str): Raw caption text.

    Returns:
        str: Cleaned caption with special tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"


def load_and_clean_captions(filepath: str) -> Dict[str, List[str]]:
    """Loads and cleans captions from a CSV file.

    Reads captions associated with image filenames, processes them with
    `clean_caption()`, and groups them by image ID.

    Args:
        filepath (str): Path to CSV file with image-caption pairs.

    Returns:
        Dict[str, List[str]]: Mapping from image filenames to cleaned captions.
    """
    captions = defaultdict(list)
    with open(filepath, "r") as file:
        next(file)  # Skip header
        for line in file:
            tokens = line.strip().split(",")
            if len(tokens) != 2:
                continue
            image_id, caption = tokens
            image_filename = image_id.split("#")[0].strip()
            cleaned = clean_caption(caption)
            captions[image_filename].append(cleaned)
    return dict(captions)


def count_word_frequencies(captions_dict: Dict[str, List[str]]) -> Counter:
    """Counts word frequencies from a dictionary of captions.

    Args:
        captions_dict (Dict[str, List[str]]): Cleaned captions by image ID.

    Returns:
        Counter: Word frequency count.
    """
    counter = Counter()
    for captions in captions_dict.values():
        for caption in captions:
            counter.update(caption.split())
    return counter


def filter_captions_by_frequency(
    captions_dict: Dict[str, List[str]], min_word_freq: int
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Replaces infrequent words with the OOV token in all captions.

    Args:
        captions_dict (Dict[str, List[str]]): Captions by image ID.
        min_word_freq (int): Frequency threshold for keeping words.

    Returns:
        Tuple[Dict[str, List[str]], Set[str]]:
            - Filtered captions with rare words replaced.
            - Set of retained vocabulary.
    """
    freq = count_word_frequencies(captions_dict)
    vocab = {
        word
        for word, count in freq.items()
        if count >= min_word_freq or word in {START_TOKEN, END_TOKEN, OOV_TOKEN}
    }

    filtered = {}
    for img_id, captions in captions_dict.items():
        new_captions = []
        for caption in captions:
            tokens = [word if word in vocab else OOV_TOKEN for word in caption.split()]
            new_captions.append(" ".join(tokens))
        filtered[img_id] = new_captions

    return filtered, vocab


def fit_tokenizer(
    filtered_captions: Dict[str, List[str]],
    num_words: int = None,
    oov_token: str = OOV_TOKEN,
) -> Tokenizer:
    """Fits a Keras Tokenizer on filtered captions.

    Args:
        filtered_captions (Dict[str, List[str]]): Cleaned captions by image ID.
        num_words (int): Maximum vocabulary size.
        oov_token (str): Token for out-of-vocabulary words.

    Returns:
        Tokenizer: Fitted tokenizer instance.
    """
    all_captions = [
        caption for captions in filtered_captions.values() for caption in captions
    ]
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters="")
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def captions_to_sequences(
    filtered_captions: Dict[str, List[str]], tokenizer: Tokenizer
) -> Dict[str, List[List[int]]]:
    """Converts tokenized captions into sequences of token IDs.

    Args:
        filtered_captions (Dict[str, List[str]]): Cleaned captions.
        tokenizer (Tokenizer): Fitted tokenizer.

    Returns:
        Dict[str, List[List[int]]]: Captions as lists of integers.
    """
    seq_dict = {}
    for img_id, captions in filtered_captions.items():
        seq_dict[img_id] = tokenizer.texts_to_sequences(captions)
    return seq_dict


def pad_caption_sequences(
    seq_dict: Dict[str, List[List[int]]], max_length: int
) -> Dict[str, List[List[int]]]:
    """Pads or truncates all caption sequences to a fixed length.

    Args:
        seq_dict (Dict[str, List[List[int]]]): Sequences of token IDs.
        max_length (int): Max sequence length.

    Returns:
        Dict[str, List[List[int]]]: Padded sequences.
    """
    padded_dict = {}
    for img_id, sequences in seq_dict.items():
        padded = pad_sequences(
            sequences, maxlen=max_length, padding="post", truncating="post"
        ).tolist()
        padded_dict[img_id] = padded
    return padded_dict


def compute_max_caption_length(
    seq_dict: Dict[str, List[List[int]]], quantile: float = 0.95
) -> int:
    """Computes maximum caption length using a quantile threshold.

    Args:
        seq_dict (Dict[str, List[List[int]]]): Sequences of token IDs.
        quantile (float): Percentile for length threshold.

    Returns:
        int: Max sequence length for padding.
    """
    lengths = [len(seq) for seqs in seq_dict.values() for seq in seqs]
    return int(np.quantile(lengths, quantile))


def save_padded_sequences(
    padded_dict: Dict[str, List[List[int]]], filepath: str, overwrite: bool = False
) -> None:
    """Saves padded sequences to a compressed `.npz` file.

    Args:
        padded_dict (Dict[str, List[List[int]]]): Padded sequences.
        filepath (str): Destination `.npz` path.
        overwrite (bool): If False, skips saving if file exists. Defaults to False.
    """
    if not overwrite and os.path.exists(filepath):
        print(f"[INFO] File already exists and overwrite=False: {filepath}")
        return
    npz_dict = {
        img_id: np.array(seqs, dtype=np.int32) for img_id, seqs in padded_dict.items()
    }
    np.savez_compressed(filepath, **npz_dict)
    print(f"[INFO] Padded sequences saved to: {filepath}")


def load_padded_sequences(filepath: str) -> Dict[str, List[List[int]]]:
    """Loads padded sequences from a `.npz` file.

    Args:
        filepath (str): Path to `.npz` file.

    Returns:
        Dict[str, List[List[int]]]: Loaded padded sequences.
    """
    data = np.load(filepath, allow_pickle=True)
    return {img_id: data[img_id].tolist() for img_id in data.files}


def save_tokenizer(
    tokenizer: Tokenizer, filepath: str, overwrite: bool = False
) -> None:
    """Saves a tokenizer to either .pkl or .json format based on file extension.

    Args:
        tokenizer (Tokenizer): Fitted tokenizer to save.
        filepath (str): Destination file path (.pkl or .json).
        overwrite (bool): If False, skips saving if file exists. Defaults to False.
    """
    if not overwrite and os.path.exists(filepath):
        print(f"[INFO] File already exists and overwrite=False: {filepath}")
        return

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pkl":
        with open(filepath, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"[INFO] Tokenizer saved to pickle file: {filepath}")
    elif ext == ".json":
        tokenizer_json = tokenizer.to_json()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(tokenizer_json)
        print(f"[INFO] Tokenizer saved to JSON file: {filepath}")
    else:
        raise ValueError("Unsupported file format. Use .pkl or .json")


def load_tokenizer(filepath: str) -> Tokenizer:
    """Loads a tokenizer from either a .pkl or .json file based on file extension.

    Args:
        filepath (str): Path to tokenizer file (.pkl or .json).

    Returns:
        Tokenizer: Loaded Keras tokenizer object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tokenizer file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pkl":
        with open(filepath, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"[INFO] Tokenizer loaded from pickle file: {filepath}")
        return tokenizer
    elif ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            tokenizer_json_str = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json_str)
        print(f"[INFO] Tokenizer loaded from JSON file: {filepath}")
        return tokenizer
    else:
        raise ValueError("Unsupported file format. Use .pkl or .json")


def build_raw_captions_dict_from_csv(captions_csv_path: str) -> Dict[str, List[str]]:
    """
    Reads a CSV file and builds a dictionary mapping image IDs to lists of captions.

    Args:
        captions_csv_path (str): Path to the CSV file with columns [image_id, caption].

    Returns:
        Dict[str, List[str]]: A dictionary mapping image_id to a list of its captions.
    """
    captions_dict = defaultdict(list)

    with open(captions_csv_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) != 2:
                continue  # Skip malformed rows
            image_id, caption = row
            captions_dict[image_id].append(caption.strip())

    return dict(captions_dict)


def load_captions_dict(captions_path: str) -> Dict[str, List[str]]:
    """
    Loads a JSON file of cleaned captions and returns a dictionary
    mapping image_id to a list of reference captions.

    Args:
        captions_path (str): Path to the JSON file containing captions.

    Returns:
        Dict[str, List[str]]: Dictionary of image_id -> list of captions.
    """
    with open(captions_path, "r") as f:
        captions_data = json.load(f)

    return captions_data

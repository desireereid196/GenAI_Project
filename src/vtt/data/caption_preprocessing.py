"""
caption_preprocessing.py

This module provides utilities for preprocessing image captions for tasks like
image captioning. It includes functions for cleaning, tokenizing, filtering,
padding, and saving/loading captions in various formats.

"""

import os
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
from vtt.utils.config import START_TOKEN, END_TOKEN, OOV_TOKEN


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
    """Saves tokenizer as a pickle file.

    Args:
        tokenizer (Tokenizer): Fitted tokenizer.
        filepath (str): Path to save the tokenizer.
        overwrite (bool): If False, skips saving if file exists. Defaults to False.
    """
    if not overwrite and os.path.exists(filepath):
        print(f"[INFO] File already exists and overwrite=False: {filepath}")
        return
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] Tokenizer saved to: {filepath}")


def load_tokenizer(filepath: str) -> Tokenizer:
    """Loads a tokenizer from a pickle file.

    Args:
        filepath (str): Path to tokenizer file.

    Returns:
        Tokenizer: Loaded tokenizer instance.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_tokenizer_json(tokenizer: Tokenizer, filepath: str) -> None:
    """Saves tokenizer in JSON format.

    Args:
        tokenizer (Tokenizer): Fitted tokenizer.
        filepath (str): Destination JSON file path.
    """
    tokenizer_json = tokenizer.to_json()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)


def load_tokenizer_json(filepath: str) -> Tokenizer:
    """Loads tokenizer from a JSON file.

    Args:
        filepath (str): Path to tokenizer JSON file.

    Returns:
        Tokenizer: Reconstructed tokenizer object.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    return tokenizer_from_json(tokenizer_json)

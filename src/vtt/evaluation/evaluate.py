"""
evaluate.py

This module provides utilities to evaluate image captioning models
using standard NLP metrics.

Supports:
- BLEU-1 to BLEU-4
- METEOR
- BERTScore

Usage:
1. evaluate_captions(): Evaluate predictions vs. ground truth captions.
2. evaluate_model(): Generate predictions for a test set and evaluate.
"""

from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import tqdm
import logging

from vtt.evaluation.metrics import (
    compute_bertscore,
    compute_bleu_scores,
    compute_meteor_scores,
)
from vtt.models.predict import generate_caption_greedy, generate_caption_beam


# Configure module-specific logger
logger = logging.getLogger(__name__)


def evaluate_captions(
    references_dict: Dict[str, List[str]], predictions_dict: Dict[str, str]
) -> Dict[str, float]:
    """
    Evaluate generated captions against reference captions using BLEU, METEOR, and BERTScore.

    Args:
        references_dict (Dict[str, List[str]]): Ground-truth captions by image ID.
        predictions_dict (Dict[str, str]): Predicted captions by image ID.

    Returns:
        Dict[str, float]: Averaged scores for each metric.
    """
    # Find common image IDs
    common_keys = list(set(references_dict.keys()) & set(predictions_dict.keys()))
    if not common_keys:
        raise ValueError("No overlapping image IDs between references and predictions.")

    # Tokenize
    tokenized_references = [
        [ref.split() for ref in references_dict[img_id]] for img_id in common_keys
    ]
    tokenized_candidates = [predictions_dict[img_id].split() for img_id in common_keys]
    raw_candidates = [predictions_dict[k] for k in common_keys]
    single_refs_for_bertscore = [references_dict[k][0] for k in common_keys]

    scores = {}
    scores.update(compute_bleu_scores(tokenized_references, tokenized_candidates))
    scores["METEOR"] = compute_meteor_scores(
        [references_dict[k] for k in common_keys], raw_candidates
    )
    scores.update(compute_bertscore(single_refs_for_bertscore, raw_candidates))

    return scores


def _evaluate_model_with_generator(
    model: Model,
    tokenizer: Tokenizer,
    features: Dict[str, np.ndarray],
    test_dataset: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray, List[str]]],
    references_dict: Dict[str, List[str]],
    max_len: int,
    caption_generator_fn: Callable[[Model, Tokenizer, np.ndarray, int], str],
) -> Dict[str, float]:
    """Generic evaluation function that uses a provided caption generation function."""
    predictions_dict = {}

    if isinstance(test_dataset, tf.data.Dataset):
        try:
            total_batches = tf.data.experimental.cardinality(test_dataset).numpy()
            if total_batches == tf.data.UNKNOWN_CARDINALITY:
                total_batches = None
        except Exception:
            total_batches = None

        tqdm_bar = tqdm.tqdm(test_dataset, total=total_batches, desc="Evaluating")

        for batch in tqdm_bar:
            (input_features_tuple, _) = batch
            (img_tensor, _, image_ids) = input_features_tuple

            for i in range(len(image_ids)):
                image_id = image_ids[i].numpy().decode("utf-8")
                image_feature = img_tensor[i].numpy()
                caption = caption_generator_fn(model, tokenizer, image_feature, max_len)
                predictions_dict[image_id] = caption
    else:
        _, _, image_ids = test_dataset
        for image_id in image_ids:
            image_feature = features[image_id]
            caption = caption_generator_fn(model, tokenizer, image_feature, max_len)
            predictions_dict[image_id] = caption

    return evaluate_captions(references_dict, predictions_dict)


def evaluate_model_greedy(
    model: Model,
    tokenizer: Tokenizer,
    features: Dict[str, np.ndarray],
    test_dataset: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray, List[str]]],
    references_dict: Dict[str, List[str]],
    max_len: int,
) -> Dict[str, float]:
    return _evaluate_model_with_generator(
        model=model,
        tokenizer=tokenizer,
        features=features,
        test_dataset=test_dataset,
        references_dict=references_dict,
        max_len=max_len,
        caption_generator_fn=generate_caption_greedy,
    )


def evaluate_model_beam(
    model: Model,
    tokenizer: Tokenizer,
    features: Dict[str, np.ndarray],
    test_dataset: Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray, List[str]]],
    references_dict: Dict[str, List[str]],
    max_len: int,
    beam_width: int = 5,
) -> Dict[str, float]:
    return _evaluate_model_with_generator(
        model=model,
        tokenizer=tokenizer,
        features=features,
        test_dataset=test_dataset,
        references_dict=references_dict,
        max_len=max_len,
        caption_generator_fn=lambda m, t, f, l: generate_caption_beam(
            m, t, f, l, beam_width=beam_width
        ),
    )

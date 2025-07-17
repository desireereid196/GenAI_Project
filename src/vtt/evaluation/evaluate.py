"""
evaluate.py

This module provides a unified interface to evaluate image captioning models
against ground-truth captions using common automatic metrics.

Supports:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- BERTScore

Usage:
    1. Provide two dictionaries:
        - references_dict: {image_id: [ref1, ref2, ...]}
        - predictions_dict: {image_id: "generated caption"}
    2. Call evaluate_captions()

Returns:
    Dictionary of average scores across the dataset.
"""

from typing import Dict, List

from vtt.evaluation.metrics import (
    compute_bertscore,
    compute_bleu_scores,
    compute_meteor_scores,
)


def evaluate_captions(
    references_dict: Dict[str, List[str]], predictions_dict: Dict[str, str]
) -> Dict[str, float]:
    """
    Evaluate generated captions against reference captions using BLEU, METEOR, and
    BERTScore.

    Args:
        references_dict (Dict[str, List[str]]): Mapping from image ID to a list of
            reference captions.
        predictions_dict (Dict[str, str]): Mapping from image ID to generated caption.

    Returns:
        Dict[str, float]: Dictionary of metric names to averaged scores.
    """
    # Find the common set of image IDs
    common_keys = list(set(references_dict.keys()) & set(predictions_dict.keys()))
    if not common_keys:
        raise ValueError(
            "No overlapping image IDs found between references and predictions."
        )

    # Tokenize references and predictions, aligned by common_keys
    tokenized_references = [
        [ref.split() for ref in references_dict[img_id]] for img_id in common_keys
    ]
    tokenized_candidates = [predictions_dict[img_id].split() for img_id in common_keys]

    # Raw format needed for METEOR and BERTScore
    raw_candidates = [predictions_dict[k] for k in common_keys]
    single_refs_for_bertscore = [references_dict[k][0] for k in common_keys]

    # Compute and aggregate scores
    scores = {}
    scores.update(compute_bleu_scores(tokenized_references, tokenized_candidates))
    scores["METEOR"] = compute_meteor_scores(
        [references_dict[k] for k in common_keys], raw_candidates
    )
    scores.update(compute_bertscore(single_refs_for_bertscore, raw_candidates))

    return scores

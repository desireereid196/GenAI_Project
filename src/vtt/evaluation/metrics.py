# vtt/evaluation/metrics.py

"""
metrics.py

This module implements core evaluation metrics for image captioning tasks.

Supported Metrics:
- BLEU-1 to BLEU-4
- METEOR
- BERTScore

Each function takes model-generated captions and corresponding ground-truth
references and returns a score or dictionary of scores.

Dependencies:
- nltk
- bert-score
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from typing import List, Dict
import bert_score
import numpy as np


def ensure_nltk_resources():
    """
    Ensure necessary NLTK resources are downloaded:
    - 'punkt' for tokenization
    - 'wordnet' for METEOR synonyms
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


# Ensure NLTK resources are ready
ensure_nltk_resources()


def compute_bleu_scores(references, candidates):
    """
    Compute BLEU-1 to BLEU-4 scores.

    Args:
        references (List[List[List[str]]]): Tokenized references for each image.
        candidates (List[List[str]]): Tokenized generated captions for each image.

    Returns:
        Dict[str, float]: Dictionary of BLEU-n scores.
    """
    smooth = SmoothingFunction().method1

    scores = {
        "BLEU-1": np.mean(
            [
                sentence_bleu(
                    ref, cand, weights=(1, 0, 0, 0), smoothing_function=smooth
                )
                for ref, cand in zip(references, candidates)
            ]
        ),
        "BLEU-2": np.mean(
            [
                sentence_bleu(
                    ref, cand, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth
                )
                for ref, cand in zip(references, candidates)
            ]
        ),
        "BLEU-3": np.mean(
            [
                sentence_bleu(
                    ref, cand, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth
                )
                for ref, cand in zip(references, candidates)
            ]
        ),
        "BLEU-4": np.mean(
            [
                sentence_bleu(
                    ref,
                    cand,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth,
                )
                for ref, cand in zip(references, candidates)
            ]
        ),
    }
    return scores


def compute_meteor_scores(references: List[List[str]], candidates: List[str]) -> float:
    """
    Compute average METEOR score over all examples.

    Args:
        references (List[List[str]]): A list of reference caption lists per image.
        candidates (List[str]): A list of generated captions per image.

    Returns:
        float: Mean METEOR score across all samples.
    """
    return np.mean(
        [
            meteor_score(
                [ref.split() for ref in refs],  # Tokenize each reference
                cand.split(),  # Tokenize candidate
            )
            for refs, cand in zip(references, candidates)
        ]
    )


def compute_bertscore(
    references: List[str], candidates: List[str], lang: str = "en"
) -> Dict[str, float]:
    """
    Compute BERTScore metrics (Precision, Recall, F1).

    Args:
        references (List[str]): List of reference captions (one per example).
        candidates (List[str]): List of generated captions.
        lang (str): Language code (default: "en").

    Returns:
        Dict[str, float]: Dictionary with BERTScore_P, BERTScore_R, BERTScore_F1.
    """
    P, R, F1 = bert_score.score(candidates, references, lang=lang, verbose=False)
    return {
        "BERTScore_P": float(P.mean()),
        "BERTScore_R": float(R.mean()),
        "BERTScore_F1": float(F1.mean()),
    }

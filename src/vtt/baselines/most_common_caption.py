"""
most_common_caption.py

Module for generating captions by assigning the single most frequently occurring
caption from a given training dataset to all test images. This serves as a
simple, non-learning baseline for evaluating image captioning models.

This baseline highlights the impact of common phrases or general descriptions
in the dataset on evaluation metrics, especially those that reward lexical
overlap (like BLEU-1). It does not consider any image content.

Functions:
    generate_most_common_caption: Determines the most frequent caption
                                  and assigns it uniformly to all test images.
"""

# Standard library imports
from collections import Counter
from typing import Dict, List
import logging # Import the logging module

# Configure module-specific logger
logger = logging.getLogger(__name__)


def generate_most_common_caption(
    test_image_ids: List[str], training_captions_pool: List[str]
) -> Dict[str, str]:
    """
    Assigns the single most common caption from the training pool to all test images.

    This baseline provides insight into how a model performs if it simply
    predicts the most statistically frequent caption, disregarding the image content.
    It can be a surprisingly strong baseline for certain metrics due to the
    prevalence of common phrases in image captioning datasets.

    Args:
        test_image_ids (List[str]): A list of unique identifiers for the test images.
                                    Each image in this list will receive the most common caption.
        training_captions_pool (List[str]): A flat list containing all individual
                                            (cleaned) captions extracted from the
                                            training dataset. This list is used to
                                            determine the most frequent caption.

    Returns:
        Dict[str, str]: A dictionary where keys are test image IDs (str) and values are
                        the single most common caption string (str). Each test image ID
                        will have this identical caption assigned.

    Raises:
        ValueError: If `training_captions_pool` is empty, as no most common caption can be determined.
    """
    logger.debug("Determining and assigning most common caption...")
    if not training_captions_pool:
        logger.error("The 'training_captions_pool' cannot be empty. Cannot determine a most common caption.")
        raise ValueError("The 'training_captions_pool' cannot be empty. Cannot determine a most common caption.")
    if not test_image_ids:
        logger.debug("No test images provided. Returning empty dictionary.")
        return {}

    caption_counts = Counter(training_captions_pool)
    most_common_caption = caption_counts.most_common(1)[0][0]

    logger.debug(f"Most common caption identified: '{most_common_caption}' (count: {caption_counts.most_common(1)[0][1]})")
    
    most_common_assignments = {
        image_id: most_common_caption for image_id in test_image_ids
    }

    logger.debug("Most common caption assignment complete.")
    return most_common_assignments
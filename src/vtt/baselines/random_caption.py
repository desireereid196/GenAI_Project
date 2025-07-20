"""
random_caption.py

Module for generating captions based on a purely random selection from a
given pool of training captions. This serves as a fundamental
non-learning baseline for evaluating image captioning models,
demonstrating the performance of a system with no understanding
of image content or linguistic structure beyond the source pool.

The caption assignment is performed with replacement and is vectorized
using NumPy for efficient processing of large datasets.

Functions:
    generate_random_captions: Assigns a random caption to each test image ID.
"""

# Standard library imports
import numpy as np
from typing import Dict, List
import logging  # Import the logging module

# Configure module-specific logger
logger = logging.getLogger(__name__)


def generate_random_captions(
    test_image_ids: List[str], training_captions_pool: List[str]
) -> Dict[str, str]:
    """
    Assigns a random caption from a pool of training captions to each test image ID.
    The selection is performed with replacement and is vectorized for efficiency.

    This baseline model provides a lower bound for performance, illustrating
    the results of a captioning system that operates purely by chance without
    any semantic or contextual understanding of the images.

    Args:
        test_image_ids (List[str]): A list of unique identifiers for the test images.
                                    Each image in this list will receive one assigned caption.
        training_captions_pool (List[str]): A flat list containing all individual
                                            (cleaned) captions extracted from the
                                            training dataset. This list serves as the
                                            source pool from which captions are randomly
                                            selected for the test images.

    Returns:
        Dict[str, str]: A dictionary where keys are test image IDs (str) and values are
                        the randomly assigned caption strings (str). Each test image ID
                        will have exactly one assigned caption.

    Raises:
        ValueError: If `training_captions_pool` is empty, as no captions can be assigned.
    """
    logger.debug("Generating random captions...")
    if not training_captions_pool:
        logger.error(
            "The 'training_captions_pool' cannot be empty. Cannot assign random captions without a source pool."
        )
        raise ValueError(
            "The 'training_captions_pool' cannot be empty. Cannot assign random captions without a source pool."
        )
    if not test_image_ids:
        logger.debug("No test images provided. Returning empty dictionary.")
        return {}  # Return empty if no test images

    captions_pool_np = np.array(training_captions_pool)
    num_captions_in_pool = len(captions_pool_np)
    num_test_images = len(test_image_ids)

    logger.debug(
        f"Assigning {num_test_images} random captions from a pool of {num_captions_in_pool}."
    )

    random_indices = np.random.choice(
        num_captions_in_pool, size=num_test_images, replace=True
    )

    assigned_captions = captions_pool_np[random_indices]
    random_assignment = dict(zip(test_image_ids, assigned_captions))

    logger.debug("Random caption generation complete.")
    return random_assignment

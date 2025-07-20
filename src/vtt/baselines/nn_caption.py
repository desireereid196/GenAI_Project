"""
nn_caption.py

Module for generating captions based on a Nearest Neighbor (NN) approach
in the image feature space. This baseline finds the closest training image(s)
to a given test image using their visual features and then assigns one of
the captions associated with those training images.

This model serves as an important non-learning baseline that assesses
the power of image feature similarity for the captioning task. It demonstrates
what can be achieved by simply reusing captions from visually similar images,
without explicit sequence generation or language modeling.

Functions:
    generate_nearest_neighbor_captions: Finds nearest neighbors in feature space
                                        and assigns corresponding captions.
"""

# Standard library imports
import numpy as np
from typing import Dict, List, Tuple
import random
import logging

# Third-party imports
from scipy.spatial.distance import cdist  # For efficient distance calculations

# Configure module-specific logger
logger = logging.getLogger(__name__)


def generate_nearest_neighbor_captions(
    test_image_ids: List[str],
    train_image_ids: List[str],
    train_image_features: Dict[str, np.ndarray],
    test_image_features: Dict[str, np.ndarray],
    train_captions_dict: Dict[str, List[str]],
    k_neighbors: int = 1,
    distance_metric: str = "cosine",
) -> Dict[str, str]:
    """
    Generates captions for test images by finding their nearest neighbor(s)
    in the training image feature space and randomly selecting a caption
    from the neighbor's associated captions.

    Args:
        test_image_ids (List[str]): A list of unique identifiers for the test images
                                    for which captions are to be generated.
        train_image_ids (List[str]): A list of unique identifiers for the training images.
                                     These correspond to the keys in `train_image_features`.
        train_image_features (Dict[str, np.ndarray]): A dictionary mapping UNIQUE
                                                      training image IDs to their
                                                      single feature vector (np.ndarray).
        test_image_features (Dict[str, np.ndarray]): A dictionary mapping UNIQUE
                                                     test image IDs to their
                                                     single feature vector (np.ndarray).
        train_captions_dict (Dict[str, List[str]]): A dictionary where keys are UNIQUE
                                                    training image IDs and values are
                                                    a list of their associated cleaned captions.
        k_neighbors (int): The number of nearest neighbors to consider for each test image.
                           If `k_neighbors > 1`, captions from all `k` nearest neighbors
                           are pooled, and one caption is randomly selected from this combined pool.
                           Defaults to 1 (finds the single closest neighbor).
        distance_metric (str): The distance metric to use for comparing image features.
                               Common choices include 'cosine' (for similarity) or 'euclidean'.

    Returns:
        Dict[str, str]: A dictionary where keys are test image IDs (str) and values are
                        the generated captions (str).

    Raises:
        ValueError: If input feature dictionaries or caption dictionary are empty,
                    or if `k_neighbors` is invalid.
    """
    if not test_image_ids or not train_image_ids:
        logger.warning(
            "Test or training image ID lists are empty. Returning empty assignments."
        )
        return {}
    if not train_image_features or not test_image_features:
        raise ValueError("Image feature dictionaries cannot be empty.")
    if not train_captions_dict:
        raise ValueError("Training captions dictionary cannot be empty.")
    if not (isinstance(k_neighbors, int) and k_neighbors >= 1):
        raise ValueError("k_neighbors must be an integer and at least 1.")

    # Prepare feature matrices for efficient distance calculation
    # Ensure features are in the same order as their IDs for matrix creation
    train_features_matrix = np.array(
        [train_image_features[img_id] for img_id in train_image_ids]
    )
    test_features_matrix = np.array(
        [test_image_features[img_id] for img_id in test_image_ids]
    )

    if train_features_matrix.size == 0 or test_features_matrix.size == 0:
        logger.warning(
            "Empty feature matrices after preparation. Returning empty assignments."
        )
        return {}

    # Calculate distances between all test image features and all training image features
    distances = cdist(
        test_features_matrix, train_features_matrix, metric=distance_metric
    )

    nearest_neighbor_assignments = {}

    for i, test_img_id in enumerate(test_image_ids):
        # Get indices of the k nearest neighbors for the current test image.
        # np.argsort returns the indices that would sort the array.
        # We take the first `k_neighbors` indices, which correspond to the smallest distances.
        nearest_indices = np.argsort(distances[i])[:k_neighbors]

        # Get the IDs of the actual nearest neighbor training images
        nearest_neighbor_train_ids = [train_image_ids[idx] for idx in nearest_indices]

        # Collect all captions from the identified nearest neighbor(s)
        candidate_captions_for_test_img = []
        for nn_id in nearest_neighbor_train_ids:
            if nn_id in train_captions_dict:
                candidate_captions_for_test_img.extend(train_captions_dict[nn_id])
            else:
                logger.warning(
                    f"Nearest neighbor '{nn_id}' not found in 'train_captions_dict'. Skipping its captions."
                )

        if candidate_captions_for_test_img:
            # Randomly choose one caption from the collected candidates.
            # This ensures if multiple neighbors or multiple captions per neighbor,
            # one is picked consistently.
            chosen_caption = random.choice(candidate_captions_for_test_img)
        else:
            # Fallback if no captions could be found for any nearest neighbor.
            # This can happen if a nearest neighbor image ID is present in features
            # but has no corresponding captions in `train_captions_dict`.
            logger.warning(
                f"Could not find any captions for test image '{test_img_id}'s nearest neighbors. Assigning empty string."
            )
            chosen_caption = ""

        nearest_neighbor_assignments[test_img_id] = chosen_caption

    return nearest_neighbor_assignments

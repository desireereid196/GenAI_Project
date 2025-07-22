"""
image_preprocessing.py

This module provides functionalities for preprocessing images and extracting
features using a pre-trained ResNet50 model. It also includes utilities
for saving and loading these extracted features to/from compressed NumPy `.npz` files.
"""

import os

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import logging

from vtt.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Configure module-specific logger
logger = logging.getLogger(__name__)


def preprocess_image(img_path: str) -> np.ndarray:
    """Preprocesses a single image for ResNet50 input.

    Loads an image, resizes it to IMAGE_SIZE, converts it to a NumPy array,
    normalizes pixel values to [0, 1], and then standardizes them using
    ImageNet mean and standard deviation. Finally, it adds a batch dimension.

    Args:
        img_path (str): The full path to the image file.

    Returns:
        np.ndarray: A preprocessed image tensor ready for model prediction,
                    with shape (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).
    """
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)  # Standardize
    return np.expand_dims(x, axis=0)  # Add batch dimension


def extract_features_from_filenames(
    image_dir: str, image_names: list[str]
) -> dict[str, np.ndarray]:
    """Extracts features from a list of image files using a pre-trained ResNet50 model.

    Initializes a ResNet50 model (pre-trained on ImageNet) and extracts features
    from its 'avg_pool' layer. It iterates through the provided image names,
    preprocesses each image, predicts its feature vector, and stores it in a
    dictionary. Images that cause an error during processing will be skipped.

    Args:
        image_dir (str): The path to the directory containing the images.
        image_names (list[str]): A list of image filenames (e.g., 'image1.jpg')
                                 relative to `image_dir`.

    Returns:
        dict[str, np.ndarray]: A dictionary where keys are image filenames
                               and values are the extracted 2048-dimensional
                               feature vectors (from ResNet50's avg_pool).
    """
    # Load ResNet50 model pre-trained on ImageNet
    model = ResNet50(weights="imagenet")
    # Create a new model that outputs the features from the 'avg_pool' layer
    extractor = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)

    features = {}
    # Iterate through image names with a progress bar
    for name in tqdm(image_names, desc="Extracting image features"):
        try:
            # Preprocess the image to prepare it for the model
            tensor = preprocess_image(os.path.join(image_dir, name))
            # Predict the feature vector (verbose=0 suppresses prediction progress bar)
            vector = extractor.predict(tensor, verbose=0)
            # Remove the batch dimension and store the feature vector
            features[name] = vector.squeeze()
        except Exception as e:
            # Print an error message and skip the problematic image
            print(f"Skipping {name}: {e}")
    return features


def extract_features_from_directory(image_dir: str) -> dict[str, np.ndarray]:
    """
    Extracts features for all images in a given directory using ResNet50.

    This function automatically identifies all `.jpg`, `.jpeg`, or `.png` images
    in the directory, preprocesses them, and extracts 2048-dim feature vectors
    using a pre-trained ResNet50 model (avg_pool layer).

    Args:
        image_dir (str): Path to the directory containing image files.

    Returns:
        dict[str, np.ndarray]: Dictionary mapping image filenames to feature vectors.
    """
    # Supported image extensions
    valid_extensions = (".jpg", ".jpeg", ".png")

    # List all valid image files in the directory
    image_names = [
        fname
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(valid_extensions)
    ]

    print(f"[INFO] Found {len(image_names)} image(s) in '{image_dir}'.")

    # Use existing function to extract features
    return extract_features_from_filenames(image_dir, image_names)


def extract_features_in_batches(
    image_dir: str,
    batch_size: int,
    output_dir: str,
    prefix: str = "features_batch",
    skip_existing: bool = True,
) -> None:
    """
    Extracts image features in batches and saves each batch to a separate .npz file.

    This function is useful when processing large datasets, as it allows you
    to incrementally save progress and recover from errors or interruptions.

    Args:
        image_dir (str): Directory containing the image files.
        batch_size (int): Number of images to process in each batch.
        output_dir (str): Directory to save batch .npz files.
        prefix (str): Filename prefix for the saved batches.
        skip_existing (bool): If True, will skip batches that already exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_names = sorted(os.listdir(image_dir))
    total_images = len(image_names)
    total_batches = int(np.ceil(total_images / batch_size))

    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_images)
        batch_image_names = image_names[batch_start:batch_end]

        batch_filename = f"{prefix}_{batch_idx:03d}.npz"
        batch_path = os.path.join(output_dir, batch_filename)

        if skip_existing and os.path.exists(batch_path):
            tqdm.write(f"[INFO] Skipping existing batch: {batch_filename}")
            continue

        tqdm.write(
            f"[INFO] Processing batch {batch_idx + 1} of ",
            "{total_batches} ({len(batch_image_names)} images)",
        )

        batch_features = extract_features_from_filenames(image_dir, batch_image_names)
        save_features(batch_features, batch_path)

        tqdm.write(f"[INFO] Saved batch to: {batch_path}")


def combine_feature_batches(batch_dir: str, output_path: str) -> None:
    """Combines all saved .npz feature batch files from a directory into a single file.

    This function is useful after processing images in batches to consolidate
    all the extracted features into one `.npz` file for easier downstream use.

    Args:
        batch_dir (str): Directory containing partial `.npz` feature files.
        output_path (str): Path to the final combined `.npz` file.
    """
    combined = {}
    for filename in tqdm(
        sorted(os.listdir(batch_dir)), desc="Combining feature batches"
    ):
        if filename.endswith(".npz"):
            batch_path = os.path.join(batch_dir, filename)
            try:
                batch = load_features(batch_path)
                combined.update(batch)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    save_features(combined, output_path)
    print(f"Combined {len(combined)} features into '{output_path}'.")


def save_features(features: dict, output_path: str) -> None:
    """Saves a dictionary of image features to a compressed NumPy .npz file.

    The keys of the dictionary become the keys within the .npz archive,
    and the values (feature arrays) are stored as the corresponding data.
    Using `savez_compressed` reduces file size.

    Args:
        features (dict): A dictionary where keys are identifiers (e.g., image names)
                         and values are NumPy arrays (e.g., feature vectors).
        output_path (str): The full path to the output .npz file (e.g., 'features.npz').
    """
    np.savez_compressed(output_path, **features)


def load_features(npz_path: str) -> dict[str, np.ndarray]:
    """Loads image features from a compressed NumPy .npz file.

    The function loads the .npz archive and reconstructs the dictionary
    of features, where keys are the original keys from the .npz file
    and values are the corresponding NumPy arrays.

    Args:
        npz_path (str): The full path to the .npz file containing the features.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the loaded features.
    """
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}

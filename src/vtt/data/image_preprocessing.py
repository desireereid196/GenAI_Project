"""
image_preprocessing.py

This module provides functionalities for preprocessing images and extracting
features using a pre-trained ResNet50 model. It also includes utilities
for saving and loading these extracted features to/from compressed NumPy `.npz` files.
"""

import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from vtt.utils.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


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

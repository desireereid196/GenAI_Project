"""
io.py

This module provides utilities for saving and loading trained Keras models,
their weights, and associated metadata (e.g., hyperparameters, tokenizer)
for the image captioning project. It supports both older HDF5 (.h5) and
newer Keras v3 (.keras) model formats.

Additionally, it includes functions to inspect saved model files
to determine the Keras version they were originally saved with,
aiding in compatibility checks and debugging.

Centralizing these operations ensures consistency and simplifies model deployment
and inference.

Functions:
    - save_model_assets: Saves the model (full or weights only), tokenizer, and metadata.
    - load_model_assets: Loads the model (full or weights only), tokenizer, and metadata.
    - get_keras_version_from_h5: Extracts Keras version from an .h5 model file.
    - get_keras_version_from_keras_file: Extracts Keras version from a .keras model file.
"""

import os
import json
import logging
from typing import Dict, Any, Callable, Tuple, Optional
import zipfile
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

# Configure module-specific logger
logger = logging.getLogger(__name__)


def save_model_assets(
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    metadata: Dict[str, Any],
    filepath_prefix: str,
    save_full_model: bool = False,
    model_format: str = "h5",
    overwrite: bool = False,
) -> None:
    """
    Saves the trained Keras model (or just its weights), the tokenizer,
    and a dictionary of associated metadata to disk.

    All files will be saved with the same `filepath_prefix` but different extensions.

    Args:
        model (tf.keras.Model): The trained Keras model to save.
        tokenizer (Tokenizer): The fitted Keras Tokenizer used for preprocessing captions.
        metadata (Dict[str, Any]): A dictionary containing all necessary metadata
                                   (e.g., `max_caption_len`, `vocab_size`, `embedding_dim`,
                                   `feature_dim`, `lstm_units`, training history, etc.).
        filepath_prefix (str): The base path and name for all saved files.
                               Example: '/path/to/my_model' will save files like
                               'my_model_weights.h5', 'my_model_tokenizer.json',
                               'my_model_metadata.json', and optionally 'my_model.h5' or 'my_model.keras'.
        save_full_model (bool): If True, saves the entire model (architecture + weights + optimizer state)
                                as a .h5 or .keras file. If False, only weights are saved.
                                Defaults to False.
        model_format (str): The format to use for saving the full model.
                            Can be 'h5' (HDF5) or 'keras' (Keras v3 format).
                            Defaults to 'h5'.
        overwrite (bool): If True, existing files will be overwritten. Defaults to False.

    Raises:
        ValueError: If an unsupported `model_format` is provided.
        Exception: If any error occurs during the saving process.
    """
    logger.info(f"Attempting to save model assets with prefix: {filepath_prefix}")

    if model_format not in ["h5", "keras"]:
        raise ValueError(
            f"Unsupported model_format: '{model_format}'. Must be 'h5' or 'keras'."
        )

    output_dir = os.path.dirname(filepath_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Save Model Weights
    weights_path = f"{filepath_prefix}_weights.weights.h5"
    if not overwrite and os.path.exists(weights_path):
        logger.warning(
            f"Weights file already exists and overwrite=False: {weights_path}. Skipping weights save."
        )
    else:
        try:
            model.save_weights(weights_path)
            logger.info(f"Model weights saved to: {weights_path}")
        except Exception as e:
            logger.error(f"Error saving model weights to {weights_path}: {e}")
            raise

    # 2. Optionally Save Full Model
    if save_full_model:
        full_model_path = f"{filepath_prefix}.{model_format}"
        if not overwrite and os.path.exists(full_model_path):
            logger.warning(
                f"Full model file already exists and overwrite=False: {full_model_path}. Skipping full model save."
            )
        else:
            try:
                model.save(full_model_path)
                logger.info(f"Full model saved to: {full_model_path}")
            except Exception as e:
                logger.error(f"Error saving full model to {full_model_path}: {e}")
                raise

    # 3. Save Tokenizer
    tokenizer_path = f"{filepath_prefix}_tokenizer.json"
    if not overwrite and os.path.exists(tokenizer_path):
        logger.warning(
            f"Tokenizer file already exists and overwrite=False: {tokenizer_path}. Skipping tokenizer save."
        )
    else:
        try:
            tokenizer_json = tokenizer.to_json()
            with open(tokenizer_path, "w", encoding="utf-8") as f:
                f.write(tokenizer_json)
            logger.info(f"Tokenizer saved to: {tokenizer_path}")
        except Exception as e:
            logger.error(f"Error saving tokenizer to {tokenizer_path}: {e}")
            raise

    # 4. Save Metadata
    metadata_path = f"{filepath_prefix}_metadata.json"
    if not overwrite and os.path.exists(metadata_path):
        logger.warning(
            f"Metadata file already exists and overwrite=False: {metadata_path}. Skipping metadata save."
        )
    else:
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}")
            raise

    logger.info("All model assets saved successfully.")


def load_model_assets(
    filepath_prefix: str,
    build_model_func: Optional[Callable[..., tf.keras.Model]] = None,
) -> Tuple[tf.keras.Model, Tokenizer, Dict[str, Any]]:
    """
    Loads a Keras model (either full model or by building and loading weights),
    its associated tokenizer, and metadata.

    It first attempts to load a full model from a .keras file, then a .h5 file.
    If neither is found, it falls back to loading weights if `build_model_func` is provided.

    Args:
        filepath_prefix (str): The base path and name of the saved files.
                               Example: '/path/to/my_model'.
        build_model_func (Optional[Callable[..., tf.keras.Model]]): A function
                                                                     that takes model
                                                                     hyperparameters
                                                                     (from metadata)
                                                                     and returns a compiled
                                                                     Keras model.
                                                                     Required if loading
                                                                     from weights only.
                                                                     Defaults to None.

    Returns:
        Tuple[tf.keras.Model, Tokenizer, Dict[str, Any]]: A tuple containing:
            - loaded_model (tf.keras.Model): The loaded Keras model.
            - loaded_tokenizer (Tokenizer): The loaded Keras Tokenizer.
            - loaded_metadata (Dict[str, Any]): The loaded metadata dictionary.

    Raises:
        FileNotFoundError: If any required file is not found.
        ValueError: If `build_model_func` is not provided when loading from weights.
        Exception: For other errors during loading.
    """
    logger.info(f"Attempting to load model assets with prefix: {filepath_prefix}")

    # 1. Load Metadata
    metadata_path = f"{filepath_prefix}_metadata.json"
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            loaded_metadata = json.load(f)
        logger.info(f"Metadata loaded from: {metadata_path}")
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        raise

    # 2. Load Tokenizer
    tokenizer_path = f"{filepath_prefix}_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer file not found: {tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    try:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_json_str = f.read()
        loaded_tokenizer = tokenizer_from_json(tokenizer_json_str)
        logger.info(f"Tokenizer loaded from: {tokenizer_path}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {tokenizer_path}: {e}")
        raise

    # 3. Load Model (prefer .keras, then .h5, then weights)
    loaded_model = None
    possible_full_model_paths = [
        f"{filepath_prefix}.keras",  # Try new .keras format first
        f"{filepath_prefix}.h5",  # Then try older .h5 format
    ]
    weights_path = f"{filepath_prefix}_weights.h5"

    for full_model_path in possible_full_model_paths:
        if os.path.exists(full_model_path):
            logger.info(f"Attempting to load full model from: {full_model_path}")
            try:
                loaded_model = tf.keras.models.load_model(full_model_path)
                logger.info("Full model loaded successfully.")
                break  # Exit loop once model is loaded
            except Exception as e:
                logger.warning(
                    f"Error loading full model from {full_model_path}: {e}. Trying next format or weights."
                )
                loaded_model = None  # Reset loaded_model if failed

    # If full model couldn't be loaded, try loading from weights
    if loaded_model is None:
        if not os.path.exists(weights_path):
            logger.error(
                f"Neither full model file nor weights file ({weights_path}) found."
            )
            raise FileNotFoundError(
                f"No model files found with prefix: {filepath_prefix}"
            )

        logger.info(f"Loading model from weights: {weights_path}")
        if build_model_func is None:
            raise ValueError(
                "`build_model_func` must be provided to load model from weights."
            )

        # Reconstruct model using metadata and load weights
        logger.info("Reconstructing model from metadata and loading weights...")
        try:
            # Use .get with default values for robustness in case metadata schema changes
            loaded_model = build_model_func(
                vocab_size=loaded_metadata["vocab_size"],
                max_caption_len=loaded_metadata["max_caption_len"],
                embedding_dim=loaded_metadata.get("embedding_dim", 256),
                feature_dim=loaded_metadata.get("feature_dim", 2048),
                lstm_units=loaded_metadata.get("lstm_units", 512),
            )
            loaded_model.load_weights(weights_path)
            logger.info("Model reconstructed and weights loaded successfully.")
        except Exception as e:
            logger.error(
                f"Error reconstructing model or loading weights from {weights_path}: {e}"
            )
            raise

    logger.info("All model assets loaded successfully.")
    return loaded_model, loaded_tokenizer, loaded_metadata


def get_keras_version_from_h5(model_path: str) -> str:
    """
    Attempts to extract the Keras version from an .h5 model file's attributes.

    Args:
        model_path (str): The full path to the .h5 model file.

    Returns:
        str: A string indicating the Keras version found, or a message if not found/error.
    """
    logger.info(f"Attempting to get Keras version from .h5 file: {model_path}")
    if not os.path.exists(model_path):
        return f"File not found: {model_path}"

    try:
        with h5py.File(model_path, "r") as f:
            # Keras 2.x models often have 'keras_version' in the root attributes
            keras_version = f.attrs.get("keras_version")
            if keras_version:
                logger.info(f"Keras version found in .h5 attributes: {keras_version}")
                return str(keras_version)  # Ensure it's a string

            # For models saved with TensorFlow's Keras, it might be under 'model_config'
            if "model_config" in f.attrs:
                model_config_str = f.attrs["model_config"]
                if isinstance(model_config_str, bytes):
                    model_config_str = model_config_str.decode("utf-8")

                try:
                    model_config = json.loads(model_config_str)
                    if "keras_version" in model_config:
                        version = model_config["keras_version"]
                        logger.info(
                            f"Keras version found in model_config (JSON) within .h5: {version}"
                        )
                        return str(version)
                except json.JSONDecodeError:
                    logger.debug(f"model_config in {model_path} is not valid JSON.")
                    pass  # Not a JSON string or malformed

            logger.info("Keras version not explicitly found in .h5 file attributes.")
            return "Keras version not explicitly found in .h5 file attributes."
    except Exception as e:
        logger.error(f"Error reading .h5 file {model_path}: {e}")
        return f"Error reading .h5 file: {e}"


def get_keras_version_from_keras_file(model_path: str) -> str:
    """
    Attempts to extract the Keras version from a .keras model file.

    Args:
        model_path (str): The full path to the .keras model file.

    Returns:
        str: A string indicating the Keras version found, or a message if not found/error.
    """
    logger.info(f"Attempting to get Keras version from .keras file: {model_path}")
    if not os.path.exists(model_path):
        return f"File not found: {model_path}"

    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            if "config.json" in zf.namelist():
                with zf.open("config.json") as f:
                    config_data = json.load(f)
                    # The Keras version is usually under 'keras_version' or 'version'
                    if "keras_version" in config_data:
                        version = config_data["keras_version"]
                        logger.info(
                            f"Keras version found in .keras config.json: {version}"
                        )
                        return str(version)
                    elif "version" in config_data:  # Sometimes just 'version'
                        version = config_data["version"]
                        logger.info(
                            f"Keras version found in .keras config.json (general version key): {version}"
                        )
                        return str(version)
                    else:
                        logger.info(
                            "Keras version not explicitly found in .keras config.json."
                        )
                        return (
                            "Keras version not explicitly found in .keras config.json."
                        )
            else:
                logger.warning("config.json not found inside .keras file.")
                return "config.json not found inside .keras file."
    except zipfile.BadZipFile:
        logger.error(f"File '{model_path}' is not a valid .keras (zip) archive.")
        return f"File '{model_path}' is not a valid .keras (zip) archive."
    except Exception as e:
        logger.error(f"Error reading .keras file {model_path}: {e}")
        return f"Error reading .keras file: {e}"

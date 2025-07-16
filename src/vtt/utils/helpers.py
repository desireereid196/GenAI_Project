"""
helpers.py

Utility functions for environment and runtime setup across the project.

This module includes tools for detecting available hardware (GPU or CPU)
and configuring TensorFlow's memory usage behavior accordingly.

Functionality:
    - Detects if TensorFlow was built with CUDA GPU support.
    - Enables memory growth to prevent full preallocation of GPU memory.
    - Fallbacks safely to CPU if no GPU is available or setup fails.
"""

import os
import random
import numpy as np
import tensorflow as tf


def detect_and_set_device() -> str:
    """
    Detect and configure TensorFlow device (GPU or CPU).

    This function checks if TensorFlow has been built with CUDA support
    and whether a GPU device is available. If a GPU is found, it attempts
    to enable memory growth to allow TensorFlow to allocate GPU memory on demand
    instead of reserving all memory at once.

    Returns:
        str: Either 'GPU' if a GPU was successfully detected and configured,
             or 'CPU' if no GPU was found or configuration failed.
    """
    # Check whether TensorFlow was compiled with GPU support
    if tf.test.is_built_with_cuda():
        # List available GPU devices
        physical_devices = tf.config.list_physical_devices("GPU")

        if physical_devices:
            print("GPU is available. Attempting to use GPU.")
            try:
                # Enable memory growth to avoid full GPU memory preallocation
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Successfully configured GPU(s): {physical_devices}")
                return "GPU"
            except RuntimeError as e:
                # RuntimeError can occur if memory growth is set after device initialization
                print(f"Error setting GPU memory growth: {e}. Falling back to CPU.")
                return "CPU"
        else:
            print(
                "No GPU devices found despite TensorFlow being built with CUDA. Using CPU."
            )
            return "CPU"
    else:
        print("TensorFlow is not built with CUDA support. Using CPU.")
        return "CPU"


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set to {seed}")

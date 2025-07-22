# src/emolex/utils.py

import tensorflow as tf

def detect_and_set_device() -> str:
    """
    Detects if a GPU is available and configures TensorFlow to use it,
    setting memory growth to avoid allocating all GPU memory at once.

    If a GPU is not found or cannot be configured, it defaults to using the CPU.

    Returns:
        str: A string indicating the detected and configured device ('GPU' or 'CPU').
    """
    # Check if TensorFlow was built with CUDA support
    if tf.test.is_built_with_cuda():
        physical_devices = tf.config.list_physical_devices('GPU')
        
        # Check if any physical GPU devices are detected
        if len(physical_devices) > 0:
            print("GPU is available. Attempting to use GPU.")
            try:
                # Set memory growth for all detected GPUs
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Successfully configured GPU: {physical_devices}")
                return 'GPU'
            except RuntimeError as e:
                # Catch errors related to memory growth configuration
                print(f"Error setting GPU memory growth: {e}. Falling back to CPU.")
                return 'CPU'
        else:
            print("No GPU devices found despite TensorFlow being built with CUDA. Using CPU.")
            return 'CPU'
    else:
        print("TensorFlow is not built with CUDA support. Using CPU.")
        return 'CPU'
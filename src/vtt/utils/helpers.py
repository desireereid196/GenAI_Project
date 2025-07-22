"""
helpers.py

Collection of general utility functions for the VTT (Visual-Text Transformer) project.

This module provides diverse helper functionalities including:
    - Environment setup and TensorFlow device configuration (GPU/CPU detection, memory growth).
    - Setting random seeds for reproducibility across different libraries.
    - Image manipulation, specifically annotating images with text captions and saving them.

Functions:
    - detect_and_set_device: Configures TensorFlow for GPU or CPU usage.
    - set_seed: Sets random seeds for reproducibility.
    - save_image_with_caption: Adds text to an image and saves it to a file.
"""

import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import logging

# Configure module-specific logger
logger = logging.getLogger(__name__)


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
    logger.info("Detecting and configuring TensorFlow device...")
    if tf.test.is_built_with_cuda():
        physical_devices = tf.config.list_physical_devices("GPU")

        if physical_devices:
            logger.info("GPU is available. Attempting to use GPU.")
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Successfully configured GPU(s): {physical_devices}")
                return "GPU"
            except RuntimeError as e:
                logger.error(
                    f"Error setting GPU memory growth: {e}. Falling back to CPU."
                )
                return "CPU"
        else:
            logger.warning(
                "No GPU devices found despite TensorFlow "
                "being built with CUDA. Using CPU."
            )
            return "CPU"
    else:
        logger.warning("TensorFlow is not built with CUDA support. Using CPU.")
        return "CPU"


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    logger.info(f"Setting random seed to {seed} for reproducibility.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seed set to {seed} across Python, NumPy, and TensorFlow.")


def save_image_with_caption(
    image_path: str,
    caption: str,
    output_path: str,
    font_size: int = 24,
    border_radius: int = 10,
):
    """
    Loads an image, adds the generated caption as black text over a white rounded rectangle,
    and saves the new image.

    This function uses PIL's default font for simplicity and cross-platform compatibility.
    The white background rectangle ensures the caption is always readable.

    Args:
        image_path (str): Full path to the original image file.
        caption (str): The generated caption text.
        output_path (str): Full path where the annotated image will be saved.
        font_size (int): Base size for the font. Note: For default font, this
                         might not directly control pixel size but serves as a
                         relative scaling factor or reference for layout.
        border_radius (int): The radius for the rounded corners of the background rectangle.
                             Defaults to 10 pixels.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        font = ImageFont.load_default()
        logger.debug("Using PIL's default font for image annotation.")

        text_color = (0, 0, 0)  # Black text
        rect_color = (255, 255, 255)  # White rectangle
        padding = 10  # Padding around the text and rectangle

        # Calculate text size and position
        # textbbox returns (left, top, right, bottom) relative to (0,0) of the text
        # We use a dummy (0,0) as the anchor for calculation, then adjust for actual position.
        # The `font_size` argument to `textbbox` is used to scale the default font.
        bbox = draw.textbbox(
            (0, 0), caption, font=font, anchor="lt", font_size=font_size
        )
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Determine the top-left corner of the text
        # Position the text at the bottom-left of the image with padding
        text_x = padding
        text_y = img.height - text_height - padding

        # Determine the rectangle coordinates
        # It should cover the text plus padding on all sides
        rect_x1 = text_x - padding
        rect_y1 = text_y - padding
        rect_x2 = text_x + text_width + padding
        rect_y2 = text_y + text_height + padding

        # Ensure rectangle doesn't go out of image bounds (especially for small images)
        rect_x1 = max(0, rect_x1)
        rect_y1 = max(0, rect_y1)
        rect_x2 = min(img.width, rect_x2)
        rect_y2 = min(img.height, rect_y2)

        # --- Draw the white ROUNDED rectangle first ---
        draw.rounded_rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2],
            radius=border_radius,  # <--- NEW: Specify border radius
            fill=rect_color,
        )

        # --- Then draw the black text on top ---
        draw.text(
            (text_x, text_y), caption, font=font, fill=text_color, font_size=font_size
        )

        img.save(output_path)
        logger.debug(f"Annotated image saved to: {output_path}")
    except Exception as e:
        logger.error(
            f"Failed to save annotated image {image_path} to {output_path}: {e}"
        )

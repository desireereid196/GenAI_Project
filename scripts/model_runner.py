"""
model_runner.py

This script orchestrates the end-to-end image captioning model pipeline.
It loads preprocessed image features and captions, loads a pretrained generative model,
runs inference on a SPECIFIC BATCH of sample images from a designated directory,
and saves the generated captions as text files AND as annotated images
to a specified output directory.

This serves as a demonstration of the integrated system as per Milestone 3 requirements,
focusing on a fixed set of input images and visual output.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import List

from vtt.data.caption_preprocessing import (
    load_tokenizer,
    load_padded_sequences,
)
from vtt.data.image_preprocessing import load_features
from vtt.models.decoder import build_decoder_model
from vtt.models.predict import generate_caption_greedy
from vtt.utils.helpers import save_image_with_caption
from vtt.data.data_loader import load_split_datasets


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("--- Starting Model Pipeline Execution ---")


# Main Execution Logic
def run_model_pipeline():
    # Path Setup
    logger.info("Setting up data paths...")

    dataset_name = "flickr8k"
    project_root = os.getcwd()  # Assumes script is run from project root

    # Directories and Files
    processed_data_dir = os.path.join(project_root, "data", "processed")
    raw_data_dir = os.path.join(project_root, "data", "raw")
    output_dir = os.path.join(project_root, "outputs", "model_runner_outputs")
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(processed_data_dir, f"{dataset_name}_features.npz")
    captions_path = os.path.join(
        processed_data_dir, f"{dataset_name}_padded_caption_sequences.npz"
    )
    tokenizer_path = os.path.join(processed_data_dir, f"{dataset_name}_tokenizer.json")
    sample_images_dir = os.path.join(raw_data_dir, "flickr8k_images_sample")
    model_weights_path = os.path.join(
        project_root, "models", "flickr8k_decoder_weights.weights.h5"
    )
    generated_captions_text_file = os.path.join(
        output_dir, "generated_sample_captions.txt"
    )

    # Load Tokenizer and Features
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.num_words

    logger.info(f"Loading pre-extracted image features from: {features_path}")
    features_dict = load_features(features_path)

    # Get sample image filenames
    sample_image_filenames = sorted(
        [
            f
            for f in os.listdir(sample_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))
        ]
    )
    if not sample_image_filenames:
        logger.error(f"No images found in directory: {sample_images_dir}")
    else:
        logger.info(f"Found {len(sample_image_filenames)} sample images to caption.")

    # Determine Max Caption Length
    logger.info("Determining maximum caption length from training data...")
    train_ds, _, _ = load_split_datasets(
        features_path=features_path,
        captions_path=captions_path,
        batch_size=64,
        val_split=0.15,
        test_split=0.10,
        shuffle=True,
        buffer_size=1000,
        seed=42,
        cache=True,
        return_numpy=False,
    )
    # Get caption shape from a batch
    for _, input_caption, _ in train_ds.take(1):
        max_caption_len = input_caption.shape[1]
        logger.info(f"Maximum caption length determined: {max_caption_len}")

    # Build and Load Model
    logger.info("Building decoder model...")
    model = build_decoder_model(vocab_size=vocab_size, max_caption_len=max_caption_len)

    logger.info(f"Loading model weights from: {model_weights_path}")
    try:
        model.load_weights(model_weights_path)
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        raise

    # Run Inference
    logger.info("Generating captions for sample images...")

    generated_captions_for_text_file: List[str] = []

    for i, image_filename in enumerate(sample_image_filenames, start=1):
        if image_filename not in features_dict:
            logger.warning(f"Features for '{image_filename}' not found. Skipping.")
            continue

        image_feature = features_dict[image_filename]
        generated_caption = generate_caption_greedy(
            model, tokenizer, image_feature, max_caption_len
        )

        # Log partial caption
        logger.info(
            f"[{i}/{len(sample_image_filenames)}] {image_filename}: {generated_caption[:60]}..."
        )

        # Save text version
        generated_captions_for_text_file.append(
            f"Image Filename: {image_filename}\nGenerated Caption: {generated_caption}\n"
        )

        # Save annotated image
        original_image_path = os.path.join(sample_images_dir, image_filename)
        image_id = os.path.splitext(image_filename)[0]
        annotated_image_path = os.path.join(output_dir, f"{image_id}_captioned.jpg")

        save_image_with_caption(
            original_image_path,
            generated_caption,
            annotated_image_path,
        )

    # Save Captions to Text File
    logger.info(f"Saving all generated captions to: {generated_captions_text_file}")
    try:
        with open(generated_captions_text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(generated_captions_for_text_file))
        logger.info("Captions successfully saved.")
    except Exception as e:
        logger.error(f"Failed to save captions to text file: {e}")
        raise

    # Print Summary
    logger.info("--- Model Pipeline Execution Finished ---")
    logger.info(f"Generated captions saved to: {generated_captions_text_file}")
    logger.info(f"Annotated images saved to: {output_dir}")


if __name__ == "__main__":
    run_model_pipeline()

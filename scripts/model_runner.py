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
from typing import List, Dict
import numpy as np
import tensorflow as tf

from vtt.data.caption_preprocessing import load_tokenizer
from vtt.data.image_preprocessing import load_features
from vtt.data.data_loader import load_split_datasets
from vtt.models.decoder import build_decoder_model
from vtt.models.predict import generate_caption_greedy
from vtt.utils.helpers import save_image_with_caption

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Configuration ---
DATASET_NAME = "flickr8k"
SAMPLE_IMAGES_SUBDIR = "flickr8k_images_sample"


# --- Main Pipeline ---
def run_model_pipeline():
    logger.info("--- Starting Model Pipeline Execution ---")
    project_root = os.getcwd()
    paths = load_paths_and_configs(project_root)

    logger.info("Loading model and tokenizer...")
    model, tokenizer, max_len = load_model_and_tokenizer(paths)

    logger.info("Loading features...")
    features_dict = load_features(paths["features"])

    logger.info("Loading sample image filenames...")
    image_filenames = get_sample_image_filenames(paths["sample_images"])

    logger.info("Generating and saving captions...")
    generate_and_save_captions(
        model=model,
        tokenizer=tokenizer,
        max_len=max_len,
        features_dict=features_dict,
        image_filenames=image_filenames,
        image_dir=paths["sample_images"],
        output_dir=paths["output_dir"],
        text_output_file=paths["output_text_file"],
    )

    logger.info("--- Model Pipeline Execution Finished ---")


def load_paths_and_configs(project_root: str) -> Dict[str, str]:
    processed = os.path.join(project_root, "data", "processed")
    raw = os.path.join(project_root, "data", "raw")
    model_dir = os.path.join(project_root, "models")
    output_dir = os.path.join(project_root, "outputs", "model_runner_outputs")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "features": os.path.join(processed, f"{DATASET_NAME}_features.npz"),
        "captions": os.path.join(
            processed, f"{DATASET_NAME}_padded_caption_sequences.npz"
        ),
        "tokenizer": os.path.join(processed, f"{DATASET_NAME}_tokenizer.json"),
        "sample_images": os.path.join(raw, SAMPLE_IMAGES_SUBDIR),
        "model_weights": os.path.join(
            model_dir, f"{DATASET_NAME}_decoder_weights.weights.h5"
        ),
        "output_dir": output_dir,
        "output_text_file": os.path.join(output_dir, "generated_sample_captions.txt"),
    }


def load_model_and_tokenizer(paths: Dict[str, str]) -> tuple:
    tokenizer = load_tokenizer(paths["tokenizer"])
    vocab_size = tokenizer.num_words

    train_ds, _, _ = load_split_datasets(
        features_path=paths["features"],
        captions_path=paths["captions"],
        batch_size=64,
        val_split=0.15,
        test_split=0.10,
        shuffle=True,
        buffer_size=1000,
        seed=42,
        cache=True,
        return_numpy=False,
    )
    for _, input_caption, _ in train_ds.take(1):
        max_len = input_caption.shape[1]
        break

    model = build_decoder_model(vocab_size=vocab_size, max_caption_len=max_len)
    model.load_weights(paths["model_weights"])

    return model, tokenizer, max_len


def get_sample_image_filenames(image_dir: str) -> List[str]:
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Sample image directory not found: {image_dir}")

    filenames = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))
        ]
    )

    if not filenames:
        raise FileNotFoundError(f"No image files found in directory: {image_dir}")

    return filenames


def generate_and_save_captions(
    model,
    tokenizer,
    max_len,
    features_dict,
    image_filenames,
    image_dir,
    output_dir,
    text_output_file,
):
    captions_text_lines = []

    for idx, filename in enumerate(image_filenames, start=1):
        if filename not in features_dict:
            logger.warning(f"Skipping '{filename}' – features not found.")
            continue

        image_feature = features_dict[filename]
        caption = generate_caption_greedy(model, tokenizer, image_feature, max_len)

        logger.info(f"[{idx}/{len(image_filenames)}] {filename}: {caption[:60]}...")

        # Save captioned image
        image_path = os.path.join(image_dir, filename)
        output_image_path = os.path.join(
            output_dir, f"{os.path.splitext(filename)[0]}_captioned.jpg"
        )
        save_image_with_caption(image_path, caption, output_image_path)

        captions_text_lines.append(
            f"Image Filename: {filename}\nGenerated Caption: {caption}\n"
        )

    # Save all captions to text file
    with open(text_output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(captions_text_lines))

    logger.info(f"Saved {len(captions_text_lines)} captions to: {text_output_file}")
    logger.info(f"Saved annotated images to: {output_dir}")


if __name__ == "__main__":
    run_model_pipeline()

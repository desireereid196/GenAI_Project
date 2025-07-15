"""
train.py

This module defines the training routine for the image captioning decoder model.

It supports:
- Training with TensorFlow's `tf.data.Dataset`
- Saving checkpoints of the best model based on training loss
- Early stopping to avoid overfitting or wasted computation

Typical usage:
    history = train_model(dataset, model, epochs=20, checkpoint_path="models/decoder.ckpt")
"""

import os
from typing import Optional
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def train_model(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int = 10,
    checkpoint_path: str = "model_checkpoint.weights.h5",
    early_stop_patience: Optional[int] = 5
) -> tf.keras.callbacks.History:
    """
    Trains the image captioning model using the provided dataset.

    This function takes a preprocessed and batched dataset, builds the model 
    by forcing a dummy forward pass (to avoid tracing errors related to variable 
    creation in tf.function), and fits the model with optional early stopping 
    and model checkpointing.

    Args:
        dataset (tf.data.Dataset): A TensorFlow dataset yielding
            ((image_features_tensor, input_caption_tensor), target_caption_tensor)
            for sequence-to-sequence training.
        model (tf.keras.Model): A compiled Keras model (e.g., ImageCaptionDecoder).
        epochs (int): Number of training epochs to run. Defaults to 10.
        checkpoint_path (str): Path to save model weights during training.
        early_stop_patience (int, optional): Number of epochs to wait for improvement
            in loss before stopping early. Set to None to disable early stopping.

    Returns:
        tf.keras.callbacks.History: A Keras History object containing training metrics.
    """
    # Force model to build before tracing starts ----
    # This avoids runtime errors caused by variable creation during model.fit()
    for (dummy_inputs, _) in dataset.take(1):
        dummy_img, dummy_caption = dummy_inputs
        _ = model((dummy_img, dummy_caption), training=False)
        break  # Only need one batch

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=1
        )
    ]
    # Add early stopping callback if applicable
    if early_stop_patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1
        ))

    # Train model
    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return history

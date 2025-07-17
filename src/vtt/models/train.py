"""
train.py

This module defines the training routine for the image captioning decoder model.

Features:
- Trains using a TensorFlow `tf.data.Dataset`
- Supports early stopping based on validation loss
- Saves best model weights during training

Typical usage:
    history = train_model(
        dataset, model, epochs=20, checkpoint_path="models/decoder.ckpt"
    )
"""

from typing import Optional

import tensorflow as tf


def train_model(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int = 10,
    checkpoint_path: str = "model_checkpoint.weights.h5",
    early_stop_patience: Optional[int] = 5,
    val_dataset: Optional[tf.data.Dataset] = None,
) -> tf.keras.callbacks.History:
    """
    Trains the image captioning model using the provided dataset.

    This function forces a dummy forward pass to build the model before training,
    then fits the model using optional early stopping and checkpointing.

    Args:
        dataset (tf.data.Dataset): Training data yielding
            ((image_features, input_caption), target_caption) tuples.
        model (tf.keras.Model): A compiled Keras model (e.g., ImageCaptionDecoder).
        epochs (int): Number of epochs to train. Defaults to 10.
        checkpoint_path (str): Path to save the best model weights.
        early_stop_patience (Optional[int]): Number of epochs with no improvement
            before early stopping. Set to None to disable.
        val_dataset (Optional[tf.data.Dataset]): Validation data for monitoring.

    Returns:
        tf.keras.callbacks.History: Keras training history object.
    """
    # Build model by calling it once with dummy input
    for dummy_inputs, _ in dataset.take(1):
        dummy_img, dummy_caption = dummy_inputs
        model((dummy_img, dummy_caption), training=False)
        break

    # Setup callbacks
    monitor_metric = "val_loss" if val_dataset else "loss"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor=monitor_metric,
            mode="min",
            verbose=1,
        )
    ]

    if early_stop_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=early_stop_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    # Train the model
    history = model.fit(
        dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history

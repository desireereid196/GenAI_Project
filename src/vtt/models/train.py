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
    model: Model,
    epochs: int = 20,
    checkpoint_path: Optional[str] = None,
    early_stop_patience: int = 3
) -> tf.keras.callbacks.History:
    """
    Trains the image captioning model using the provided dataset and optional callbacks.

    Args:
        dataset (tf.data.Dataset): Preprocessed training dataset of the form 
                                   ((image_features, caption_inputs), caption_targets).
        model (tf.keras.Model): The compiled decoder model to be trained.
        epochs (int): Number of training epochs. Defaults to 20.
        checkpoint_path (str, optional): Path to save model checkpoint (only best by loss).
                                         If None, no checkpoints will be saved.
        early_stop_patience (int): Number of epochs to wait for loss improvement before stopping.
                                   If 0 or None, early stopping is disabled.

    Returns:
        tf.keras.callbacks.History: Training history object containing loss per epoch.
    """
    callbacks = []

    # ---- ModelCheckpoint Callback ----
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="loss",                # Only monitor training loss (no val_loss)
            save_best_only=True,          # Only save if loss improves
            save_weights_only=True,       # Save only weights (not the full model)
            verbose=1
        )
        callbacks.append(checkpoint_cb)

    # ---- EarlyStopping Callback ----
    if early_stop_patience and early_stop_patience > 0:
        early_stop_cb = EarlyStopping(
            monitor="loss",                # Stop if training loss plateaus
            patience=early_stop_patience,
            restore_best_weights=True     # Load best weights at the end
        )
        callbacks.append(early_stop_cb)

    # ---- Model Training ----
    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return history

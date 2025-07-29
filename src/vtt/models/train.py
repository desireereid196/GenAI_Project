"""
train.py

This module defines the training routine for the image captioning decoder model.

Features:
- Trains using a TensorFlow `tf.data.Dataset`
- Ignores image_id (used for evaluation only) during training
- Supports early stopping based on validation loss
- Saves best model weights during training

Typical usage:
    history = train_model(
        dataset=train_ds,
        model=decoder,
        epochs=20,
        checkpoint_path="models/decoder.ckpt",
        val_dataset=val_ds,
    )
"""

from typing import Optional
import tensorflow as tf
import logging


# Configure module-specific logger
logger = logging.getLogger(__name__)


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
        dataset (tf.data.Dataset): Training data yielding 3-tuples
            ((image_tensor, input_caption_tensor), target_caption_tensor, image_id).
        model (tf.keras.Model): A compiled Keras model (e.g., ImageCaptionDecoder).
        epochs (int): Number of epochs to train. Defaults to 10.
        checkpoint_path (str): Path to save the best model weights.
        early_stop_patience (Optional[int]): Number of epochs with no improvement
            before early stopping. Set to None to disable.
        val_dataset (Optional[tf.data.Dataset]): Validation data for monitoring.

    Returns:
        tf.keras.callbacks.History: Keras training history object.
    """

    # Dummy call to build model (eager mode)
    for dummy_inputs, _ ,_ in dataset.take(1):
        dummy_img, dummy_caption  = dummy_inputs  # Discard image_id
        model((dummy_img, dummy_caption), training=False)
        break

    # Define a function to strip image_id from each batch
    def strip_image_id(inputs, target, imgage_id):
        #image_tensor, input_caption, _ = inputs
        image_tensor, input_caption = inputs
        return (image_tensor, input_caption), target

    # Apply mapping to strip image_id from dataset
    dataset = dataset.map(strip_image_id, num_parallel_calls=tf.data.AUTOTUNE)
    if val_dataset:
        val_dataset = val_dataset.map(
            strip_image_id, num_parallel_calls=tf.data.AUTOTUNE
        )
    # Define callbacks
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

"""
vtt/models/decoder.py

This module defines the architecture for the decoder component of an image captioning
system.

The decoder receives:
    - A precomputed image embedding (e.g., from ResNet-50)
    - A sequence of tokenized caption inputs

The architecture includes:
    - Dense projection of image features
    - Word embedding of input captions
    - Concatenation of visual and linguistic inputs
    - LSTM for temporal modeling
    - Softmax output over the vocabulary

The model is trained using teacher forcing to predict the next word in the sequence.

Typical use:
    model = build_decoder_model(vocab_size, max_caption_len, feature_dim=2048) # Added feature_dim
"""

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Embedding, Input, Lambda
from tensorflow.keras.models import Model
import logging

# Configure module-specific logger
logger = logging.getLogger(__name__)


def build_decoder_model(
    vocab_size: int,
    max_caption_len: int,
    feature_dim: int = 2048,
    embedding_dim: int = 256,
    lstm_units: int = 512,
) -> tf.keras.Model:
    """
    Builds and compiles a decoder model for image captioning.

    Args:
        vocab_size (int): Size of the vocabulary used for caption generation.
        max_caption_len (int): Maximum length of the caption sequences.
        feature_dim (int): Dimensionality of the input image features (e.g., 2048 for ResNet50).
                           Defaults to 2048.
        embedding_dim (int): Dimensionality of word embeddings. Defaults to 256.
        lstm_units (int): Number of LSTM units. Defaults to 512.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    logger.info(
        f"Building decoder model with vocab_size={vocab_size}, max_caption_len={max_caption_len}, "
        f"feature_dim={feature_dim}, embedding_dim={embedding_dim}, lstm_units={lstm_units}"
    )

    # ----- Image Feature Input -----
    # Input shape: (batch_size, feature_dim), e.g., from ResNet avg_pool layer
    img_input = Input(
        shape=(feature_dim,), name="image_input"
    )  # <--- USING feature_dim HERE

    # Project image embedding to same dimension as word embeddings
    img_emb = Dense(embedding_dim, activation="relu", name="image_dense")(img_input)

    # Add time dimension to match caption input shape: (batch_size, 1, embedding_dim)
    img_emb = Lambda(lambda x: tf.expand_dims(x, 1), name="expand_image")(img_emb)

    # ----- Caption Input -----
    # Input shape: (batch_size, max_caption_len)
    caption_input = Input(shape=(max_caption_len,), name="caption_input")

    # Embed caption tokens: output shape (batch_size, max_caption_len, embedding_dim)
    caption_emb = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,  # Masking will be handled by the loss or custom training loop if needed
        name="caption_embedding",
    )(caption_input)

    # ----- Merge Inputs -----
    # Concatenate along time axis:
    # output shape (batch_size, max_caption_len+1, embedding_dim)
    merged = Concatenate(axis=1, name="merge_image_caption")([img_emb, caption_emb])

    # ----- Sequence Modeling -----
    # Apply LSTM: returns sequence of hidden states for each time step
    lstm_out = LSTM(units=lstm_units, return_sequences=True, name="lstm")(merged)

    # ----- Output Layer -----
    # Output shape: (batch_size, max_caption_len+1, vocab_size)
    output = Dense(units=vocab_size, activation="softmax", name="output_dense")(
        lstm_out
    )

    # Remove the first timestep output (image position) to align with caption target
    # The first timestep of `merged` corresponds to the image embedding.
    # The target captions start from the first actual word after <start>,
    # so we trim the output to match the target sequence length.
    output = Lambda(lambda x: x[:, 1:, :], name="trim_image_output")(output)

    # ----- Compile Model -----
    model = Model(
        inputs=[img_input, caption_input],
        outputs=output,
        name="ImageCaptionDecoder",
    )
    # Using sparse_categorical_crossentropy because targets are integer sequences
    # and output is a softmax over vocabulary.
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    logger.info("Decoder model built and compiled.")

    return model

import pickle
import tensorflow as tf
from vtt.models.decoder import build_decoder_model

def load_model_and_tokenizer(
    weights_path="models/best_model.h5",
    tokenizer_path="data/processed/flickr8k_tokenizer.pkl"
):
    """
    Rebuilds the decoder model from architecture and loads weights.
    Fixes Lambda tf.expand_dims error by ensuring tf context is available.
    """

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1
    max_caption_len = 19  # Set this to match training

    # Build model architecture
    model = build_decoder_model(vocab_size, max_caption_len)

    # Load weights
    model.load_weights(weights_path)

    return model, tokenizer

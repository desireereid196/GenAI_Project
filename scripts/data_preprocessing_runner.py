import numpy as np
from vtt.data.data_loader import load_split_datasets
from vtt.data.caption_preprocessing import *
from vtt.data.image_preprocessing import (
    extract_features_from_directory,
    save_features,
    load_features,
)
from vtt.utils.config import END_TOKEN, OOV_TOKEN, START_TOKEN

#Pre-process images
def pre_process_images(image_dir,output_features_path):
    # Extract features for all images in the directory
    features = extract_features_from_directory(image_dir)
    # Save the full dictionary of features to disk
    save_features(features, output_features_path)

#Pre-process captions
def pre_process_captions(captions_path,padded_caption_sequences_path,tokenizer_path):
    
    # Step 1: Load and clean raw captions
    captions_dict = load_and_clean_captions(captions_path)
    # Step 2: Filter out rare words and build vocabulary
    filtered_captions, vocab = filter_captions_by_frequency(captions_dict, min_word_freq=5)
    # Step 3: Fit tokenizer on filtered captions
    tokenizer = fit_tokenizer(filtered_captions, num_words=10000)
    # Step 4: Convert cleaned captions to sequences of token IDs
    seqs = captions_to_sequences(filtered_captions, tokenizer)
    # Step 5: Compute max length for padding using 95th percentile
    max_length = compute_max_caption_length(seqs, quantile=0.95)
    # Step 6: Pad all sequences to uniform length
    padded_seqs = pad_caption_sequences(seqs, max_length=max_length)
    # Step 7: Save processed data and tokenizer
    save_padded_sequences( padded_seqs, padded_caption_sequences_path)
    save_tokenizer(tokenizer,tokenizer_path)
    

#Pass processed images and captions to split data 
def get_processed_data(features_path,captions_path):
    train_ds, val_ds, test_ds = load_split_datasets(
    features_path=features_path,
    captions_path=captions_path,
    batch_size=64,
    val_split=0.15,
    test_split=0.10,
    shuffle=True,
    buffer_size=1000,
    seed=42,
    cache=True,
    return_numpy=False
    )
    return train_ds, val_ds, test_ds

if __name__ == "__main__":

    dataset_name = "flickr8k"     
    image_dir = f"../data/flickr8k_images/subset/"  
    raw_captions_path = f"../data/raw/{dataset_name}_captions.csv"
    padded_sequences_path =f"../data/processed/{dataset_name}_padded_caption_sequences.npz"
    features_path = f"../data/processed/{dataset_name}_features.npz"
    captions_path = f"../data/processed/{dataset_name}_padded_caption_sequences.npz"
    tokenizer_path = f"../data/processed/{dataset_name}_tokenizer.json"
    
    pre_process_images(image_dir,features_path)
    pre_process_captions(raw_captions_path,padded_sequences_path,tokenizer_path)

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union, List


def load_features_and_sequences(
    features_path: str, captions_path: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load aligned image features and caption sequences from .npz files,
    along with associated image IDs for each (feature, caption) pair.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: Features, captions, image IDs
    """
    features_npz = np.load(features_path)
    captions_npz = np.load(captions_path, allow_pickle=True)

    image_features = []
    caption_sequences = []
    image_ids = []

    for img_id in captions_npz.files:
        if img_id in features_npz:
            for caption in captions_npz[img_id]:
                image_features.append(features_npz[img_id])
                caption_sequences.append(caption)
                image_ids.append(img_id)

    return (
        np.asarray(image_features, dtype=np.float32),
        np.asarray(caption_sequences, dtype=np.int32),
        image_ids,
    )


def prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Split each caption into (input, target) for seq2seq training,
    while preserving image IDs.

    Input dataset should yield (image, caption, image_id).
    Output yields ((image, caption_input, image_id), caption_target).
    """

    def split_inputs_and_targets(img, caption, img_id):
        return ((img, caption[:-1], img_id), caption[1:])

    return dataset.map(split_inputs_and_targets, num_parallel_calls=tf.data.AUTOTUNE)


def load_split_datasets(
    features_path: str,
    captions_path: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    shuffle: bool = True,
    buffer_size: int = 1000,
    seed: Optional[int] = 42,
    cache: bool = False,
    return_numpy: bool = False,
) -> Union[
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    Tuple[
        np.ndarray,
        np.ndarray,
        List[str],
        np.ndarray,
        np.ndarray,
        List[str],
        np.ndarray,
        np.ndarray,
        List[str],
    ],
]:
    """
    Load and split dataset into train/val/test, preserving image IDs for evaluation.
    """
    image_features, caption_sequences, image_ids = load_features_and_sequences(
        features_path, captions_path
    )
    assert len(image_features) == len(caption_sequences) == len(image_ids)

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(image_features))
        image_features = image_features[idx]
        caption_sequences = caption_sequences[idx]
        image_ids = [image_ids[i] for i in idx]

    total = len(image_features)
    val_size = int(val_split * total)
    test_size = int(test_split * total)
    train_size = total - val_size - test_size

    train_X, train_y, train_ids = (
        image_features[:train_size],
        caption_sequences[:train_size],
        image_ids[:train_size],
    )
    val_X, val_y, val_ids = (
        image_features[train_size : train_size + val_size],
        caption_sequences[train_size : train_size + val_size],
        image_ids[train_size : train_size + val_size],
    )
    test_X, test_y, test_ids = (
        image_features[train_size + val_size :],
        caption_sequences[train_size + val_size :],
        image_ids[train_size + val_size :],
    )

    if return_numpy:
        return (
            train_X,
            train_y,
            train_ids,
            val_X,
            val_y,
            val_ids,
            test_X,
            test_y,
            test_ids,
        )

    def make_dataset(
        features: np.ndarray, captions: np.ndarray, ids: List[str], drop_remainder: bool
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((features, captions, ids))
        ds = prepare_dataset(ds)

        if cache:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size)

        return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(
            tf.data.AUTOTUNE
        )

    return (
        make_dataset(train_X, train_y, train_ids, True),
        make_dataset(val_X, val_y, val_ids, False),
        make_dataset(test_X, test_y, test_ids, False),
    )

"""
Data loading and preprocessing utilities for chest X-ray classification.

Dataset: Chest X-Ray Images (Pneumonia)
Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected directory structure after download:
    data/
        chest_xray/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
"""

import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/chest_xray")
IMG_SIZE = (224, 224)          # resize target for all models
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ── TODO 1 ───────────────────────────────────────────────────────────────────
def get_data_generators(
    data_dir: str | Path = DATA_DIR,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
) -> tuple:
    """
    Create Keras ImageDataGenerators for train, validation, and test splits.

    Tasks:
        1. Build a train generator with augmentation (horizontal_flip,
           rotation_range=10, zoom_range=0.1, width/height_shift_range=0.1)
           and rescale pixel values to [0, 1].
        2. Build val and test generators with rescaling only (no augmentation).
        3. Use flow_from_directory with:
               - class_mode='binary'
               - target_size=img_size
               - batch_size=batch_size
               - seed=SEED
        4. Return (train_gen, val_gen, test_gen).

    Args:
        data_dir:     Root folder containing train/, val/, test/ sub-folders.
        img_size:     (height, width) to resize images to.
        batch_size:   Mini-batch size.
        augment_train: Apply data augmentation to the training set.

    Returns:
        (train_generator, val_generator, test_generator)
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 1: implement get_data_generators()")
    data_dir = Path(data_dir)
    
    # Training augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            horizontal_flip=True,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255
        )
    # Validation and Test generators (no augmentation)
    val_test_datagen = ImageDataGenerator(
        rescale=1.0/255
    )
    # Train generator
    train_gen = train_datagen.flow_from_directory(
        directory=data_dir / "train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        seed=SEED
    )
    # Validation generator
    val_gen = val_test_datagen.flow_from_directory(
        directory=data_dir / "val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        seed=SEED
    )
    # Test generator
    test_gen = val_test_datagen.flow_from_directory(
        directory=data_dir / "test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        seed=SEED
    )
    
    return train_gen, val_gen, test_gen
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 2 ───────────────────────────────────────────────────────────────────
def get_tf_datasets(
    data_dir: str | Path = DATA_DIR,
    img_size: tuple = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
) -> tuple:
    """
    Alternative: build tf.data.Dataset pipelines for train, val, and test.

    Tasks:
        1. Use tf.keras.utils.image_dataset_from_directory for each split.
        2. Normalise pixel values (divide by 255.0).
        3. For the training set, optionally apply random horizontal flip and
           random rotation augmentation using tf.keras.layers.
        4. Cache, shuffle (train only), and prefetch each dataset.
        5. Return (train_ds, val_ds, test_ds).

    Args:
        data_dir:      Root folder with train/, val/, test/ sub-folders.
        img_size:      (height, width) resize target.
        batch_size:    Mini-batch size.
        augment_train: Whether to apply augmentation layers to training set.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 2: implement get_tf_datasets()")
    data_dir = Path(data_dir)
    
    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True,
        seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False,
        seed=SEED
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False,
        seed=SEED
    )
    
    # Normalisation layer
    normalization_layer = tf.keras.layers.Rescaling(1.0/255)
    
    # Data Augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1)
    ])
    
    # Normalize datasets
    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation to training set if specified
    if augment_train:
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
    # Cache, Shuffle and prefetch
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 3 ───────────────────────────────────────────────────────────────────
def compute_class_weights(train_generator) -> dict:
    """
    Compute class weights to handle class imbalance.

    The dataset has ~4,273 pneumonia vs ~1,583 normal samples.
    Weighted loss helps the model treat both classes fairly.

    Tasks:
        1. Count samples per class from train_generator.classes.
        2. Use sklearn.utils.class_weight.compute_class_weight with
           class_weight='balanced'.
        3. Return a dict mapping class index -> weight, e.g. {0: 1.35, 1: 0.74}.

    Args:
        train_generator: A Keras DirectoryIterator with a .classes attribute.

    Returns:
        dict: {class_index: weight}
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 3: implement compute_class_weights()")
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    classes = np.unique(train_generator.classes)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_generator.classes
    )
    return dict(enumerate(class_weights))
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


def describe_dataset(data_dir: str | Path = DATA_DIR) -> dict:
    """
    Return a summary dict with per-split class counts.
    Already implemented — do not modify.
    """
    data_dir = Path(data_dir)
    summary = {}
    for split in ("train", "val", "test"):
        split_path = data_dir / split
        if not split_path.exists():
            summary[split] = {}
            continue
        summary[split] = {
            cls: len(list((split_path / cls).glob("*.jpeg"))
                     + list((split_path / cls).glob("*.jpg"))
                     + list((split_path / cls).glob("*.png")))
            for cls in CLASS_NAMES
            if (split_path / cls).exists()
        }
    return summary

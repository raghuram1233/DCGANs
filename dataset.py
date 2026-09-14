"""
Data loading and preprocessing utilities for DCGAN training and evaluation.
"""

import os
from typing import Tuple, Union
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


def get_data_generator(
    data_dir: str,
    img_shape: Tuple[int, int] = (64, 64),
    batch_size: int = 128,
    horizontal_flip: bool = True,
) -> DirectoryIterator:
    """
    Creates an image data generator normalized to [-1.0, 1.0].

    Args:
        data_dir: Root directory containing images (organized in subdirectories or flat).
        img_shape: Target image size (height, width).
        batch_size: Batch size for training.
        horizontal_flip: Whether to apply random horizontal flips for data augmentation.

    Returns:
        DirectoryIterator yielding batches of images scaled to [-1, 1].
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. Please provide a valid directory containing image data."
        )

    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: (x / 127.5) - 1.0,
        horizontal_flip=horizontal_flip,
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_shape[:2],
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
    )

    return train_gen


def extract_real_images(
    data_source: Union[DirectoryIterator, np.ndarray],
    num_samples: int = 100,
) -> np.ndarray:
    """
    Extracts a fixed number of real images from an iterator or array for evaluation.

    Args:
        data_source: DirectoryIterator or numpy array containing images.
        num_samples: Number of samples to extract.

    Returns:
        Numpy array of shape (num_samples, H, W, C).
    """
    real_images = []

    if hasattr(data_source, "next") or "Iterator" in str(type(data_source)):
        if hasattr(data_source, "reset"):
            data_source.reset()

        collected = 0
        batch_idx = 0

        while collected < num_samples and batch_idx < len(data_source):
            try:
                batch_data = data_source[batch_idx]
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    batch_x, _ = batch_data
                else:
                    batch_x = batch_data

                for img in batch_x:
                    if collected >= num_samples:
                        break
                    real_images.append(img)
                    collected += 1
                batch_idx += 1
            except (StopIteration, IndexError, ValueError) as e:
                print(f"Extraction stopped at batch {batch_idx}: {e}")
                break

    elif hasattr(data_source, "shape") or isinstance(data_source, (list, tuple)):
        real_images = data_source[:num_samples]
    else:
        raise ValueError(f"Unsupported data source type: {type(data_source)}")

    if len(real_images) == 0:
        raise ValueError("No images could be extracted from the data source.")

    return np.array(real_images)

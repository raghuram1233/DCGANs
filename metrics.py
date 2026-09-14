"""
Evaluation metrics for Generative Adversarial Networks:
- Fréchet Inception Distance (FID)
- Inception Score (IS)
"""

from typing import Dict, Tuple, Union
import cv2
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input


class GANMetrics:
    """
    Computes quantitative evaluation metrics (FID and IS) for image generative models.
    """

    def __init__(self):
        print("Loading pre-trained InceptionV3 models for evaluation...")
        self.inception_features = InceptionV3(
            include_top=False,
            pooling="avg",
            input_shape=(299, 299, 3),
            weights="imagenet",
        )
        self.inception_classifier = InceptionV3(
            include_top=True,
            input_shape=(299, 299, 3),
            weights="imagenet",
        )
        print("InceptionV3 models loaded successfully.")

    def preprocess_for_inception(self, images: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        """
        Preprocesses images to shape (N, 299, 299, 3) with Inception preprocessing.
        """
        if tf.is_tensor(images):
            images = images.numpy()

        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        # Rescale if needed
        if images.max() <= 1.0:
            if images.min() < 0.0:
                images = (images + 1.0) * 127.5
            else:
                images = images * 255.0

        images = np.clip(images, 0, 255).astype(np.uint8)

        processed = []
        for img in images:
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 1):
                if len(img.shape) == 3:
                    img = img[:, :, 0]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img_resized = cv2.resize(img, (299, 299))
            processed.append(img_resized)

        processed = np.array(processed, dtype=np.float32)
        return preprocess_input(processed)

    def calculate_fid(self, real_images: np.ndarray, fake_images: np.ndarray) -> float:
        """
        Calculates Fréchet Inception Distance between real and synthetic image distributions.
        """
        real_prep = self.preprocess_for_inception(real_images)
        fake_prep = self.preprocess_for_inception(fake_images)

        real_features = self.inception_features.predict(real_prep, batch_size=32, verbose=0)
        fake_features = self.inception_features.predict(fake_prep, batch_size=32, verbose=0)

        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)

        diff = mu_real - mu_fake
        covmean = sqrtm(sigma_real @ sigma_fake)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return float(fid)

    def calculate_inception_score(
        self,
        fake_images: np.ndarray,
        splits: int = 10,
    ) -> Tuple[float, float]:
        """
        Calculates Inception Score (mean and standard deviation).
        """
        fake_prep = self.preprocess_for_inception(fake_images)
        preds = self.inception_classifier.predict(fake_prep, batch_size=32, verbose=0)

        scores = []
        n_part = len(preds) // splits

        for i in range(splits):
            start_idx = i * n_part
            end_idx = start_idx + n_part if i < splits - 1 else len(preds)
            part = preds[start_idx:end_idx]

            p_y = np.mean(part, axis=0)
            kl_divs = [entropy(p_yx, p_y) for p_yx in part]
            scores.append(np.exp(np.mean(kl_divs)))

        return float(np.mean(scores)), float(np.std(scores))


def generate_fake_images(
    generator: tf.keras.Model,
    noise_dim: int = 128,
    num_samples: int = 100,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Batched generation of fake images from random standard normal noise.
    """
    fake_images = []
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        noise = tf.random.normal([current_batch_size, noise_dim])
        batch_fake = generator(noise, training=False)
        fake_images.append(batch_fake.numpy())
    return np.concatenate(fake_images, axis=0)

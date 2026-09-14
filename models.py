"""
Core Model Architectures and Training Components for DCGAN / WGAN-GP.

Includes:
- Generator with Self-Attention (supports both UpSampling2D+Conv2D and Conv2DTranspose)
- Deep Convolutional Discriminator / Critic
- WGAN_GP model with Gradient Penalty and Instance Noise
- Training Callbacks: ResultsCallback (image grids, GIF, checkpoints) and LRScheduler
"""

import os
import cv2
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import MultiHeadAttention


def build_generator(
    noise_dim: int = 128,
    output_channels: int = 3,
    activation: str = "tanh",
    alpha: float = 0.2,
    upsampling: str = "upsample",
) -> tf.keras.Model:
    """
    Constructs the Generator model with Self-Attention at 32x32 feature maps.

    Args:
        noise_dim: Dimension of latent input vector z.
        output_channels: Number of image channels (3 for RGB).
        activation: Output layer activation function (typically 'tanh' for [-1, 1] range).
        alpha: LeakyReLU negative slope coefficient.
        upsampling: 'upsample' (UpSampling2D + Conv2D, reduces checkerboard artifacts)
                    or 'transpose' (Conv2DTranspose).

    Returns:
        Keras Model producing (batch_size, 64, 64, output_channels) images.
    """
    inputs = layers.Input(shape=(noise_dim,), name="noise_input")
    x = layers.Dense(4 * 4 * 512, use_bias=False)(inputs)
    x = layers.Reshape((4, 4, 512))(x)

    # Upsampling blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
    for filters in [512, 256, 128, 64]:
        if upsampling == "transpose":
            x = layers.Conv2DTranspose(
                filters,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
            )(x)
        else:
            # UpSampling2D + Conv2D avoids checkerboard artifacts
            x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
            x = layers.Conv2D(filters, (5, 5), padding="same", use_bias=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha)(x)

        # Self-attention at 32x32 feature map (filters=128)
        if filters == 128:
            seq = layers.Reshape((32 * 32, 128))(x)
            attn_out = MultiHeadAttention(num_heads=4, key_dim=128 // 8)(seq, seq)
            seq = layers.Add()([seq, attn_out])
            x = layers.Reshape((32, 32, 128))(seq)

        if filters == 64:
            x = layers.Dropout(0.5)(x)

    outputs = layers.Conv2D(
        output_channels,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation=activation,
        use_bias=False,
        dtype="float32",
    )(x)

    assert outputs.shape == (None, 64, 64, output_channels)
    return tf.keras.Model(inputs, outputs, name=f"generator_{upsampling}")


def build_discriminator(
    img_shape: tuple = (64, 64, 3),
    activation: str = "linear",
    alpha: float = 0.2,
) -> tf.keras.Model:
    """
    Constructs the Discriminator (Critic) model.

    Args:
        img_shape: Tuple representing (height, width, channels) of input images.
        activation: Output activation ('linear' for WGAN-GP critic score).
        alpha: LeakyReLU negative slope coefficient.

    Returns:
        Keras Model outputting a scalar score for each input image.
    """
    inputs = layers.Input(shape=img_shape, name="image_input")
    x = inputs

    # Downsampling blocks: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
        x = layers.LeakyReLU(alpha)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation=activation, dtype="float32")(x)

    return tf.keras.Model(inputs, outputs, name="discriminator")


class WGAN_GP(tf.keras.models.Model):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) and dynamic instance noise.
    """

    def __init__(
        self,
        discriminator: tf.keras.Model,
        generator: tf.keras.Model,
        noise_dim: int,
        discriminator_extra_steps: int = 5,
        gp_weight: float = 10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.discriminator_extra_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn=None, g_loss_fn=None, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn or (lambda real, fake: tf.reduce_mean(fake) - tf.reduce_mean(real))
        self.g_loss_fn = g_loss_fn or (lambda fake: -tf.reduce_mean(fake))

    def add_instance_noise(self, x, stddev: float = 0.1):
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=stddev, dtype=x.dtype)
        return x + noise

    def gradient_penalty(self, real, fake):
        batch_size = tf.shape(real)[0]
        epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interp = epsilon * real + (1.0 - epsilon) * fake
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = self.discriminator(interp, training=True)
        grads = tape.gradient(pred, interp)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, (tuple, list)):
            real_images = real_images[0]

        # Train Discriminator for discriminator_extra_steps
        d_losses, gp_vals = [], []
        for _ in range(self.discriminator_extra_steps):
            batch_size = tf.shape(real_images)[0]
            noise = tf.random.normal([batch_size, self.noise_dim])

            real_noisy = self.add_instance_noise(real_images)
            fake_images = self.generator(noise, training=True)
            fake_noisy = self.add_instance_noise(fake_images)

            with tf.GradientTape() as tape:
                d_real = self.discriminator(real_noisy, training=True)
                d_fake = self.discriminator(fake_noisy, training=True)
                gp = self.gradient_penalty(real_noisy, fake_noisy)
                d_loss = self.d_loss_fn(d_real, d_fake) + self.gp_weight * gp

            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

            d_losses.append(d_loss)
            gp_vals.append(gp)

        # Train Generator
        noise = tf.random.normal([tf.shape(real_images)[0], self.noise_dim])
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            d_fake = self.discriminator(fake_images, training=True)
            g_loss = self.g_loss_fn(d_fake)

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {
            "d_loss": tf.reduce_mean(d_losses),
            "g_loss": g_loss,
            "gp": tf.reduce_mean(gp_vals),
        }


class ResultsCallback(Callback):
    """
    Generates image grids from a fixed seed at epoch end, saves checkpoints, and compiles a GIF.
    """

    def __init__(
        self,
        noise_dim: int,
        output_path: str,
        examples: int = 16,
        grid: tuple = (4, 4),
        spacing: int = 5,
        gif_size: tuple = (416, 416),
        duration: float = 0.04,
        save_model: bool = True,
    ):
        super().__init__()
        self.seed = tf.random.normal([examples, noise_dim])
        self.results = []
        self.out_dir = os.path.join(output_path, "results")
        os.makedirs(self.out_dir, exist_ok=True)
        self.grid = grid
        self.spacing = spacing
        self.gif_size = gif_size
        self.duration = duration
        self.save_model = save_model

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.generator(self.seed, training=False)
        imgs = ((preds * 127.5) + 127.5).numpy().astype(np.uint8)

        # Construct image grid
        h, w, c = imgs.shape[1:]
        grid_img = np.zeros(
            (
                self.grid[0] * h + (self.grid[0] - 1) * self.spacing,
                self.grid[1] * w + (self.grid[1] - 1) * self.spacing,
                c,
            ),
            dtype=np.uint8,
        )
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                idx = i * self.grid[1] + j
                if idx < len(imgs):
                    grid_img[
                        i * (h + self.spacing) : i * (h + self.spacing) + h,
                        j * (w + self.spacing) : j * (w + self.spacing) + w,
                    ] = imgs[idx]

        cv2.imwrite(
            os.path.join(self.out_dir, f"img_{epoch:03d}.png"),
            cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
        )
        self.results.append(cv2.resize(grid_img, self.gif_size, interpolation=cv2.INTER_AREA))

        if self.save_model:
            mdir = os.path.join(self.out_dir, "models")
            os.makedirs(mdir, exist_ok=True)
            self.model.generator.save(os.path.join(mdir, f"gen_{epoch:03d}.h5"))
            self.model.discriminator.save(os.path.join(mdir, f"disc_{epoch:03d}.h5"))

    def on_train_end(self, logs=None):
        if self.results:
            gif_frames = [imageio.core.util.Image(img[..., ::-1]) for img in self.results]
            imageio.mimsave(
                os.path.join(self.out_dir, "output.gif"),
                gif_frames,
                duration=self.duration,
            )


class LRScheduler(Callback):
    """
    Linear learning rate decay callback with optional TensorBoard logging.
    """

    def __init__(self, decay_epochs: int, min_lr: float = 2e-6, tb=None):
        super().__init__()
        self.decay_epochs = decay_epochs
        self.min_lr = min_lr
        self.tb = tb

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.decay_epochs:
            scale = 1.0 - (epoch / self.decay_epochs)
            new_lr = max(float(self.model.g_optimizer.learning_rate.numpy()) * scale, self.min_lr)
            self.model.g_optimizer.learning_rate.assign(new_lr)
            self.model.d_optimizer.learning_rate.assign(new_lr)
            print(f"Epoch {epoch}: adjusted lr = {new_lr:.6f}")
            if self.tb:
                with self.tb._writers["train"].as_default():
                    tf.summary.scalar("learning_rate", new_lr, step=epoch)

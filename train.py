"""
Command-Line Interface for Training WGAN-GP with Self-Attention.
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import TensorBoard

from models import (
    build_generator,
    build_discriminator,
    WGAN_GP,
    ResultsCallback,
    LRScheduler,
)
from dataset import get_data_generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WGAN-GP with Multi-Head Self-Attention on Face Images."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to root directory containing training images.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/01_wgan_gp",
        help="Output directory to save results, checkpoints, and logs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["upsample", "transpose"],
        default="upsample",
        help="Generator upsampling method: 'upsample' (UpSampling2D+Conv2D, avoids checkerboard artifacts) or 'transpose' (Conv2DTranspose).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--noise_dim", type=int, default=128, help="Latent noise vector dimension.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial Adam learning rate.")
    parser.add_argument("--gp_weight", type=float, default=10.0, help="Gradient penalty weight lambda.")
    parser.add_argument(
        "--extra_d_steps",
        type=int,
        default=5,
        help="Number of discriminator updates per generator update.",
    )
    parser.add_argument(
        "--resume_gen",
        type=str,
        default=None,
        help="Optional path to pre-trained generator weights (.h5).",
    )
    parser.add_argument(
        "--resume_disc",
        type=str,
        default=None,
        help="Optional path to pre-trained discriminator weights (.h5).",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Enable mixed_float16 precision policy for training acceleration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Mixed precision setup
    if args.mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        print("Mixed precision policy 'mixed_float16' enabled.")

    print(f"Loading dataset from: {args.data_dir}")
    train_gen = get_data_generator(
        data_dir=args.data_dir,
        img_shape=(64, 64),
        batch_size=args.batch_size,
    )
    print(f"Dataset loaded: {train_gen.samples} images across {train_gen.num_classes} folders.")

    print(f"Building models (Mode: {args.mode})...")
    gen = build_generator(noise_dim=args.noise_dim, upsampling=args.mode)
    disc = build_discriminator(img_shape=(64, 64, 3))

    if args.resume_gen and os.path.isfile(args.resume_gen):
        print(f"Loading generator weights from {args.resume_gen}...")
        gen.load_weights(args.resume_gen)

    if args.resume_disc and os.path.isfile(args.resume_disc):
        print(f"Loading discriminator weights from {args.resume_disc}...")
        disc.load_weights(args.resume_disc)

    # Optimizers
    d_opt = tf.keras.optimizers.Adam(args.lr, beta_1=0.0, beta_2=0.9)
    g_opt = tf.keras.optimizers.Adam(args.lr, beta_1=0.0, beta_2=0.9)

    def d_loss_fn(real, fake):
        return tf.reduce_mean(fake) - tf.reduce_mean(real)

    def g_loss_fn(fake):
        return -tf.reduce_mean(fake)

    wgan = WGAN_GP(
        discriminator=disc,
        generator=gen,
        noise_dim=args.noise_dim,
        discriminator_extra_steps=args.extra_d_steps,
        gp_weight=args.gp_weight,
    )
    wgan.compile(d_opt, g_opt, d_loss_fn, g_loss_fn, run_eagerly=False)

    # Callbacks
    tb_log_dir = os.path.join(args.out_dir, "logs")
    tb = TensorBoard(log_dir=tb_log_dir)
    cb_results = ResultsCallback(noise_dim=args.noise_dim, output_path=args.out_dir)
    cb_lr = LRScheduler(decay_epochs=args.epochs, min_lr=2e-6, tb=tb)

    print(f"Beginning training for {args.epochs} epochs...")
    wgan.fit(
        train_gen,
        epochs=args.epochs,
        callbacks=[cb_results, tb, cb_lr],
    )
    print(f"Training completed successfully. Checkpoints and samples saved to {args.out_dir}/results.")


if __name__ == "__main__":
    main()

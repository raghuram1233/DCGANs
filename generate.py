"""
Command-Line Interface for Sample Generation and Latent Space Interpolation.
"""

import argparse
import os
import cv2
import imageio
import numpy as np
import tensorflow as tf

from models import build_generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic faces or latent walks using trained WGAN-GP generator."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained generator weights (.h5). If not supplied, uses random initialization.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["upsample", "transpose"],
        default="upsample",
        help="Generator architecture variant: 'upsample' or 'transpose'.",
    )
    parser.add_argument("--noise_dim", type=int, default=128, help="Latent noise vector dimension.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory to save images.")
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Generate a smooth latent interpolation walk GIF between two random latent vectors.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of transition frames for latent space interpolation.",
    )
    return parser.parse_args()


def slerp(val: float, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between two vectors."""
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1.0, 1.0))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def save_grid(images: np.ndarray, output_path: str, grid_rows: int = 4, grid_cols: int = 4, spacing: int = 4):
    h, w, c = images.shape[1:]
    grid_h = grid_rows * h + (grid_rows - 1) * spacing
    grid_w = grid_cols * w + (grid_cols - 1) * spacing
    grid_canvas = np.zeros((grid_h, grid_w, c), dtype=np.uint8)

    for i in range(grid_rows):
        for j in range(grid_cols):
            idx = i * grid_cols + j
            if idx < len(images):
                y = i * (h + spacing)
                x = j * (w + spacing)
                grid_canvas[y : y + h, x : x + w] = images[idx]

    cv2.imwrite(output_path, cv2.cvtColor(grid_canvas, cv2.COLOR_RGB2BGR))
    print(f"Saved sample grid to: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Instantiating generator (mode: {args.mode}, noise_dim: {args.noise_dim})...")
    gen = build_generator(noise_dim=args.noise_dim, upsampling=args.mode)

    if args.weights and os.path.isfile(args.weights):
        print(f"Loading weights from {args.weights}...")
        gen.load_weights(args.weights)
    else:
        if args.weights:
            print(f"Warning: Weights file '{args.weights}' not found. Using random weights.")
        else:
            print("No weights specified. Generating with initial weights.")

    # Generate sample grid
    z = tf.random.normal([args.num_samples, args.noise_dim])
    preds = gen(z, training=False)
    imgs = ((preds * 127.5) + 127.5).numpy().astype(np.uint8)

    grid_side = int(np.ceil(np.sqrt(args.num_samples)))
    grid_path = os.path.join(args.out_dir, "generated_samples.png")
    save_grid(imgs, grid_path, grid_rows=grid_side, grid_cols=grid_side)

    # Optional latent space walk
    if args.interpolate:
        print(f"Generating latent interpolation sequence ({args.steps} steps)...")
        z1 = np.random.randn(args.noise_dim)
        z2 = np.random.randn(args.noise_dim)

        frames = []
        for step in np.linspace(0, 1, args.steps):
            z_interp = slerp(step, z1, z2)
            z_tensor = tf.constant(z_interp.reshape(1, -1), dtype=tf.float32)
            pred = gen(z_tensor, training=False)[0]
            img = ((pred * 127.5) + 127.5).numpy().astype(np.uint8)
            frames.append(img)

        # Loop animation smoothly
        loop_frames = frames + frames[::-1]
        gif_path = os.path.join(args.out_dir, "latent_interpolation.gif")
        imageio.mimsave(gif_path, loop_frames, duration=0.04)
        print(f"Saved latent interpolation animation to: {gif_path}")


if __name__ == "__main__":
    main()

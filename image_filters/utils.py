"""Shared helper functions used by both filter scripts."""
import os
from pathlib import Path

import cv2
import numpy as np


def validate_kernel(ksize, sigma=None):
    if not isinstance(ksize, int) or ksize < 3 or ksize % 2 == 0:
        raise ValueError("Kernel size must be an odd integer >= 3")
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("Sigma must be finite and positive")


def load_gray(path):
    """Read an image from disk as a grayscale uint8 array."""
    # imdecode handles Unicode Windows paths that cv2.imread may reject.
    img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save(path, img):
    """Save an image, creating the output folder if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok, encoded = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise OSError(f"Could not save image: {path}")
    Path(path).write_bytes(encoded.tobytes())


def to_uint8(img):
    """Round to nearest integer, clip to [0, 255], and convert to uint8."""
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def normalize(img):
    """Stretch any float image to the full 0-255 range (useful for edge maps)."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.05, seed=42):
    """Randomly turn `amount` of the pixels black or white."""
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    mask = rng.random(img.shape)
    noisy[mask < amount / 2] = 0
    noisy[mask > 1 - amount / 2] = 255
    return noisy


def add_gaussian_noise(img, sigma=20, seed=42):
    """Add zero-mean Gaussian noise with standard deviation `sigma`."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    return to_uint8(img.astype(np.float32) + noise)


def show_grid(images, titles, save_path, cols=3):
    """Plot several images in a grid and save the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
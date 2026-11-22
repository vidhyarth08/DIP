"""
Smoothing (Low-Pass) Filters
============================
Mean, Weighted Average, Gaussian and Median filters.

Smoothing filters reduce noise and fine detail by replacing each pixel
with a value computed from its neighbourhood.

Usage:
    python smoothing_filters.py --input images/input.png --noise salt_pepper
"""
import argparse

import cv2
import numpy as np

from utils import (add_gaussian_noise, add_salt_pepper_noise, load_gray,
                   save, show_grid, validate_kernel)


# ----------------------------------------------------------------------
# 1. Mean (Box) Filter
# ----------------------------------------------------------------------
def mean_filter(img, ksize=5):
    """Every neighbour has the same weight: kernel = ones / (k*k)."""
    validate_kernel(ksize)
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(img, -1, kernel)


# ----------------------------------------------------------------------
# 2. Weighted Average Filter
# ----------------------------------------------------------------------
def weighted_average_filter(img):
    """The centre pixel gets the highest weight, the edges the lowest.

    Classic 3x3 kernel (sum of weights = 16):
        1 2 1
        2 4 2   / 16
        1 2 1
    """
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], np.float32) / 16.0
    return cv2.filter2D(img, -1, kernel)


# ----------------------------------------------------------------------
# 3. Gaussian Filter
# ----------------------------------------------------------------------
def gaussian_kernel(ksize=5, sigma=1.0):
    """Build a 2D Gaussian kernel (outer product of two 1D Gaussians)."""
    validate_kernel(ksize, sigma)
    ax = np.arange(ksize) - ksize // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = np.outer(g, g)
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_filter(img, ksize=5, sigma=1.5):
    """Weights fall off smoothly with distance from the centre."""
    return cv2.filter2D(img, -1, gaussian_kernel(ksize, sigma))


# ----------------------------------------------------------------------
# 4. Median Filter (non-linear)
# ----------------------------------------------------------------------
def median_filter(img, ksize=5):
    """Replace each pixel with the median of its neighbourhood.
    Excellent against salt-and-pepper noise and keeps edges sharp."""
    validate_kernel(ksize)
    return cv2.medianBlur(img, ksize)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Smoothing filters demo")
    parser.add_argument("--input", default="images/input.png")
    parser.add_argument("--output", default="output")
    parser.add_argument("--noise", choices=["none", "salt_pepper", "gaussian"],
                        default="none",
                        help="Optional noise to add before filtering (default: none)")
    parser.add_argument("--ksize", type=int, default=5,
                        help="Kernel size for mean/median/gaussian (odd number)")
    args = parser.parse_args()

    img = load_gray(args.input)

    if args.noise == "salt_pepper":
        img = add_salt_pepper_noise(img, amount=0.05)
    elif args.noise == "gaussian":
        img = add_gaussian_noise(img, sigma=20)

    results = {
        "Input" if args.noise == "none" else "Input (noisy)": img,
        f"Mean {args.ksize}x{args.ksize}": mean_filter(img, args.ksize),
        "Weighted Average 3x3": weighted_average_filter(img),
        f"Gaussian {args.ksize}x{args.ksize}": gaussian_filter(img, args.ksize),
        f"Median {args.ksize}x{args.ksize}": median_filter(img, args.ksize),
    }

    names = ["input_noisy", "mean", "weighted_average", "gaussian", "median"]
    for name, im in zip(names, results.values()):
        save(f"{args.output}/smoothing_{name}.png", im)

    show_grid(list(results.values()), list(results.keys()),
              f"{args.output}/smoothing_comparison.png", cols=3)
    print(f"Done! Results saved in '{args.output}/'")


if __name__ == "__main__":
    main()
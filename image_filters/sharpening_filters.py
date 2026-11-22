"""
Sharpening (High-Pass) Filters
==============================
Laplacian, Sobel and High-Boost filters.

Sharpening filters highlight edges and fine detail by emphasising
rapid intensity changes.

Usage:
    python sharpening_filters.py --input images/input.png
"""
import argparse

import cv2
import numpy as np

from utils import load_gray, normalize, save, show_grid, to_uint8, validate_kernel


# ----------------------------------------------------------------------
# 1. Laplacian Filter (2nd-order derivative)
# ----------------------------------------------------------------------
def laplacian_edges(img, neighbours=4):
    """Return the raw Laplacian response (edges only), scaled for display."""
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], np.float32)
    if neighbours == 8:
        kernel = np.ones((3, 3), np.float32)
        kernel[1, 1] = -8
    elif neighbours != 4:
        raise ValueError("Laplacian neighbours must be 4 or 8")
    lap = cv2.filter2D(img.astype(np.float32), -1, kernel)
    return lap, normalize(np.abs(lap))


def laplacian_sharpen(img, neighbours=4):
    """sharpened = original - Laplacian  (kernel has a negative centre)."""
    lap, _ = laplacian_edges(img, neighbours)
    return to_uint8(img.astype(np.float32) - lap)


# ----------------------------------------------------------------------
# 2. Sobel Filter (1st-order derivative / gradient)
# ----------------------------------------------------------------------
def sobel_edges(img):
    """Gradient magnitude = sqrt(Gx^2 + Gy^2)."""
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], np.float32)
    gy_kernel = gx_kernel.T
    f = img.astype(np.float32)
    gx = cv2.filter2D(f, -1, gx_kernel)
    gy = cv2.filter2D(f, -1, gy_kernel)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return magnitude, normalize(magnitude)


def sobel_sharpen(img, strength=0.25):
    """Add a fraction of the gradient magnitude back to the original."""
    magnitude, _ = sobel_edges(img)
    return to_uint8(img.astype(np.float32) + strength * magnitude)


# ----------------------------------------------------------------------
# 3. High-Boost Filter
# ----------------------------------------------------------------------
def unsharp_mask(img, k=1.0, ksize=5, sigma=1.5):
    """high_boost = original + k * (original - blurred)

    k = 1  -> classic unsharp masking
    k > 1  -> high-boost filtering (stronger sharpening)
    """
    validate_kernel(ksize, sigma)
    if not np.isfinite(k) or k < 0:
        raise ValueError("Unsharp strength must be finite and nonnegative")
    f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(f, (ksize, ksize), sigma)
    mask = f - blurred
    return to_uint8(f + k * mask)


def high_boost(img, A=1.5, ksize=5, sigma=1.5):
    """High-boost filtering: g = A*f - blurred(f), A >= 1.

    A=1 yields a signed high-pass response, clipped here for PNG display.
    A=2 equals unsharp masking with k=1. A=1.5 can darken smooth regions.
    """
    validate_kernel(ksize, sigma)
    if not np.isfinite(A) or A < 1:
        raise ValueError("High-boost A must be finite and >= 1")
    f = img.astype(np.float32)
    return to_uint8(A * f - cv2.GaussianBlur(f, (ksize, ksize), sigma))


def prewitt_edges(img):
    """Compute edge magnitude using standard 3x3 Prewitt gradients."""
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    f = img.astype(np.float32)
    gx = cv2.filter2D(f, -1, kernel)
    gy = cv2.filter2D(f, -1, kernel.T)
    magnitude = np.hypot(gx, gy)
    return magnitude, normalize(magnitude)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sharpening filters demo")
    parser.add_argument("--input", default="images/input.png")
    parser.add_argument("--output", default="output")
    parser.add_argument("--A", type=float, default=1.5,
                        help="High-boost factor A >= 1 (default 1.5)")
    args = parser.parse_args()

    img = load_gray(args.input)

    _, lap_edges = laplacian_edges(img)
    _, sob_edges = sobel_edges(img)

    results = {
        "Input": img,
        "Laplacian (edges)": lap_edges,
        "Laplacian Sharpened": laplacian_sharpen(img),
        "Sobel (edges)": sob_edges,
        "Sobel Sharpened": sobel_sharpen(img),
        f"High-Boost (A={args.A})": high_boost(img, A=args.A),
    }

    names = ["input", "laplacian_edges", "laplacian_sharpened",
             "sobel_edges", "sobel_sharpened", "high_boost"]
    for name, im in zip(names, results.values()):
        save(f"{args.output}/sharpening_{name}.png", im)

    show_grid(list(results.values()), list(results.keys()),
              f"{args.output}/sharpening_comparison.png", cols=3)
    print(f"Done! Results saved in '{args.output}/'")


if __name__ == "__main__":
    main()
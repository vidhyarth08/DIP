"""Apply all spatial filters to one common grayscale input and save comparisons."""
import argparse
import json
from pathlib import Path

from smoothing_filters import mean_filter, weighted_average_filter, gaussian_filter, median_filter
from sharpening_filters import (laplacian_edges, laplacian_sharpen, sobel_edges,
                                prewitt_edges, high_boost, unsharp_mask)
from utils import (load_gray, save, show_grid, validate_kernel,
                   add_gaussian_noise, add_salt_pepper_noise)

HERE = Path(__file__).resolve().parent


def apply_filters(img, ksize=3, sigma=1.5, A=1.5):
    """Every result is independently computed from the same image, never chained."""
    validate_kernel(ksize, sigma)
    smoothing = {
        "mean": mean_filter(img, ksize),
        "weighted_average": weighted_average_filter(img),
        "gaussian": gaussian_filter(img, ksize, sigma),
        "median": median_filter(img, ksize),
    }
    sharpening = {
        "laplacian_4_edges": laplacian_edges(img, 4)[1],
        "laplacian_4_sharpened": laplacian_sharpen(img, 4),
        "laplacian_8_edges": laplacian_edges(img, 8)[1],
        "laplacian_8_sharpened": laplacian_sharpen(img, 8),
        "sobel_edges": sobel_edges(img)[1],
        "prewitt_edges": prewitt_edges(img)[1],
        "high_boost": high_boost(img, A, ksize, sigma),
        "unsharp_mask": unsharp_mask(img, 1.0, ksize, sigma),
    }
    return {"smoothing": smoothing, "sharpening": sharpening}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "images/input.png")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--A", type=float, default=1.5, help="High-boost factor >= 1")
    parser.add_argument("--noise", choices=["none", "gaussian", "salt_pepper"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        original = load_gray(str(args.input))
        img = original.copy()
        if args.noise == "gaussian":
            img = add_gaussian_noise(img, sigma=25, seed=args.seed)
        elif args.noise == "salt_pepper":
            img = add_salt_pepper_noise(img, amount=0.03, seed=args.seed)
        groups = apply_filters(img, args.ksize, args.sigma, args.A)
        save(str(args.output / "original.png"), original)
        save(str(args.output / "input.png"), img)
        for group, results in groups.items():
            for name, result in results.items():
                save(str(args.output / group / f"{name}.png"), result)
            show_grid([img, *results.values()], ["Common input", *[n.replace('_', ' ').title() for n in results]],
                      str(args.output / f"{group}_comparison.png"), cols=3)
        metadata = {"input": args.input.name, "shape": list(img.shape), "noise": args.noise,
                    "seed": args.seed, "ksize": args.ksize, "sigma": args.sigma, "A": args.A,
                    "filters": {group: list(results) for group, results in groups.items()}}
        (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved 12 filter results and two comparison sheets to {args.output}")


if __name__ == "__main__":
    main()
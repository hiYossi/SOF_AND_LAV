"""Part A: data representation and preprocessing."""

from __future__ import annotations

import argparse

import numpy as np

from .config import (
    DATASET_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_IMAGES,
    DEFAULT_MAX_IMAGES_PER_PERSON,
    RANDOM_SEED,
)
from .data import load_dataset, load_pgm, parse_image_size, parse_max_images_per_person


def run_part_a(dataset_path=DATASET_PATH, image_size=DEFAULT_IMAGE_SIZE,
               max_images=DEFAULT_MAX_IMAGES,
               max_images_per_person=DEFAULT_MAX_IMAGES_PER_PERSON,
               random_seed=RANDOM_SEED, use_cache=True, verbose=True):
    """Load the dataset and summarize the preprocessing pipeline."""
    X, y, label_to_name, image_paths = load_dataset(
        dataset_path,
        image_size=image_size,
        max_images=max_images,
        max_images_per_person=max_images_per_person,
        cache_name="part_a",
        use_cache=use_cache,
        random_seed=random_seed,
        verbose=verbose,
    )

    sample_image = load_pgm(image_paths[0])
    processed_shape = sample_image.shape if image_size is None else image_size
    class_counts = np.bincount(y)

    summary = {
        "num_samples": int(len(X)),
        "num_classes": int(len(label_to_name)),
        "original_shape": tuple(sample_image.shape),
        "processed_shape": tuple(processed_shape),
        "feature_count": int(X.shape[1]),
        "class_count_min": int(class_counts.min()),
        "class_count_max": int(class_counts.max()),
        "preprocessing_steps": [
            (
                "Preserve the original 64x72 resolution"
                if image_size is None else
                f"Resize images to {processed_shape[0]}x{processed_shape[1]}"
            ),
            "Normalize each image independently into the [0, 1] range",
            "Flatten each image into one feature vector for the classical models",
            "Use PCA later as a learned low-dimensional representation before classification",
        ],
    }

    print("=" * 70)
    print("PART A - DATA REPRESENTATION AND PREPROCESSING")
    print("=" * 70)
    print(f"Samples loaded: {summary['num_samples']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Original image shape: {summary['original_shape']}")
    print(f"Processed image shape: {summary['processed_shape']}")
    print(f"Flattened feature count: {summary['feature_count']}")
    print(f"Class balance range: {summary['class_count_min']} to {summary['class_count_max']} samples")
    print("\nPreprocessing steps:")
    for step in summary["preprocessing_steps"]:
        print(f"  - {step}")

    return summary


def build_parser():
    """Build the Part A CLI."""
    parser = argparse.ArgumentParser(description="Run Part A preprocessing inspection.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the labeled dataset zip.")
    parser.add_argument("--image-height", type=int, default=None, help="Optional resized image height.")
    parser.add_argument("--image-width", type=int, default=None, help="Optional resized image width.")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES, help="Optional total cap on images.")
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_PERSON,
        help="Balanced cap per identity. Use 0 for all images.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset caching.")
    parser.add_argument("--quiet", action="store_true", help="Reduce loader output.")
    return parser


def main(argv=None):
    """CLI entry point for Part A."""
    args = build_parser().parse_args(argv)
    image_size = parse_image_size(args.image_height, args.image_width)
    run_part_a(
        dataset_path=args.dataset,
        image_size=image_size,
        max_images=args.max_images,
        max_images_per_person=parse_max_images_per_person(args.max_images_per_person),
        random_seed=args.seed,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )
    return 0

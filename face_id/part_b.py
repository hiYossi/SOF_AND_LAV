"""Part B: supervised model introduction and comparison."""

from __future__ import annotations

import argparse

from .config import (
    DATASET_PATH,
    DEFAULT_COMPONENT_GRID,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_IMAGES,
    DEFAULT_MAX_IMAGES_PER_PERSON,
    DEFAULT_KNN_NEIGHBORS,
    DEFAULT_SVM_EPOCHS_GRID,
    DEFAULT_SVM_LEARNING_RATE_GRID,
    DEFAULT_SVM_REG_GRID,
    DEFAULT_VALIDATION_RATIO,
    RANDOM_SEED,
)
from .data import load_dataset, parse_image_size, parse_max_images_per_person
from .model_registry import MODEL_SPECS, build_search_space, normalize_model_names
from .selection import format_hyperparams, select_hyperparameters_with_validation_split
from .splits import stratified_holdout_indices


def run_part_b(dataset_path=DATASET_PATH, model_names=None, image_size=DEFAULT_IMAGE_SIZE,
               max_images=DEFAULT_MAX_IMAGES,
               max_images_per_person=DEFAULT_MAX_IMAGES_PER_PERSON,
               validation_ratio=DEFAULT_VALIDATION_RATIO,
               component_grid=DEFAULT_COMPONENT_GRID,
               svm_reg_grid=DEFAULT_SVM_REG_GRID,
               svm_epochs_grid=DEFAULT_SVM_EPOCHS_GRID,
               svm_learning_rate_grid=DEFAULT_SVM_LEARNING_RATE_GRID,
               knn_neighbors=DEFAULT_KNN_NEIGHBORS,
               random_seed=RANDOM_SEED, use_cache=True, verbose=True):
    """Compare the supervised candidate models on one stratified validation split."""
    model_names = normalize_model_names(model_names or ["all"])
    X, y, label_to_name, _ = load_dataset(
        dataset_path,
        image_size=image_size,
        max_images=max_images,
        max_images_per_person=max_images_per_person,
        cache_name="part_b",
        use_cache=use_cache,
        random_seed=random_seed,
        verbose=verbose,
    )
    train_indices, val_indices = stratified_holdout_indices(
        y,
        holdout_ratio=validation_ratio,
        seed=random_seed,
    )
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    search_space = build_search_space(
        model_names,
        max_supported_components=min(len(X_train), X.shape[1]),
        component_grid=component_grid,
        svm_reg_grid=svm_reg_grid,
        svm_epochs_grid=svm_epochs_grid,
        svm_learning_rate_grid=svm_learning_rate_grid,
        knn_neighbors=knn_neighbors,
    )

    print("=" * 70)
    print("PART B - SUPERVISED MODELS")
    print("=" * 70)
    print(f"Samples loaded: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Candidate models: {', '.join(MODEL_SPECS[name].display_name for name in model_names)}")

    results = {}
    for model_name, hyperparameter_grid in search_space.items():
        selection = select_hyperparameters_with_validation_split(
            X_train,
            y_train,
            X_val,
            y_val,
            num_classes=len(label_to_name),
            model_name=model_name,
            hyperparameter_grid=hyperparameter_grid,
            use_pca_cache=use_cache,
        )
        results[model_name] = selection
        print(f"\n{MODEL_SPECS[model_name].display_name}")
        print(f"  Best hyperparameters: {format_hyperparams(selection['best_hyperparams'])}")
        print(f"  Train accuracy:      {selection['best_train_score']:.4f}")
        print(f"  Validation accuracy: {selection['best_validation_score']:.4f}")

    return results


def build_parser():
    """Build the Part B CLI."""
    parser = argparse.ArgumentParser(description="Run Part B supervised-model comparison.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the labeled dataset zip.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names to compare: linear_least_squares svm knn all",
    )
    parser.add_argument("--image-height", type=int, default=None, help="Optional resized image height.")
    parser.add_argument("--image-width", type=int, default=None, help="Optional resized image width.")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES, help="Optional total cap on images.")
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_PERSON,
        help="Balanced cap per identity. Use 0 for all images.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Validation fraction for the single holdout split.",
    )
    parser.add_argument(
        "--component-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_COMPONENT_GRID),
        help="PCA component counts to try.",
    )
    parser.add_argument(
        "--svm-reg-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_SVM_REG_GRID),
        help="Regularization strengths to try for the SVM.",
    )
    parser.add_argument(
        "--svm-epochs-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_SVM_EPOCHS_GRID),
        help="Epoch counts to try for the SVM.",
    )
    parser.add_argument(
        "--svm-learning-rate-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_SVM_LEARNING_RATE_GRID),
        help="Learning rates to try for the SVM.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        nargs="+",
        default=list(DEFAULT_KNN_NEIGHBORS),
        help="Neighbor counts to try for k-NN.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset and PCA caching.")
    parser.add_argument("--quiet", action="store_true", help="Reduce loader output.")
    return parser


def main(argv=None):
    """CLI entry point for Part B."""
    args = build_parser().parse_args(argv)
    image_size = parse_image_size(args.image_height, args.image_width)
    run_part_b(
        dataset_path=args.dataset,
        model_names=args.models,
        image_size=image_size,
        max_images=args.max_images,
        max_images_per_person=parse_max_images_per_person(args.max_images_per_person),
        validation_ratio=args.validation_ratio,
        component_grid=args.component_grid,
        svm_reg_grid=args.svm_reg_grid,
        svm_epochs_grid=args.svm_epochs_grid,
        svm_learning_rate_grid=args.svm_learning_rate_grid,
        knn_neighbors=args.knn_neighbors,
        random_seed=args.seed,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )
    return 0

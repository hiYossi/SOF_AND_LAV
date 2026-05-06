"""Part D: nested K-fold model selection."""

from __future__ import annotations

import argparse

from .config import (
    DATASET_PATH,
    DEFAULT_COMPONENT_GRID,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INNER_FOLDS,
    DEFAULT_KNN_NEIGHBORS,
    DEFAULT_MAX_IMAGES,
    DEFAULT_MAX_IMAGES_PER_PERSON,
    DEFAULT_OUTER_FOLDS,
    RANDOM_SEED,
)
from .data import load_dataset, parse_image_size, parse_max_images_per_person
from .model_registry import MODEL_SPECS, build_search_space, normalize_model_names
from .selection import (
    choose_best_model,
    fit_final_model,
    format_hyperparam_counter,
    format_hyperparams,
    predict_single_image_with_model,
    run_nested_k_fold_cv,
    select_hyperparameters_with_inner_cv,
    summarize_nested_cv_results,
)
from .splits import validate_nested_cv_setup


def run_part_d(dataset_path=DATASET_PATH, model_names=None, image_size=DEFAULT_IMAGE_SIZE,
               max_images=DEFAULT_MAX_IMAGES,
               max_images_per_person=DEFAULT_MAX_IMAGES_PER_PERSON,
               outer_folds=DEFAULT_OUTER_FOLDS,
               inner_folds=DEFAULT_INNER_FOLDS,
               component_grid=DEFAULT_COMPONENT_GRID,
               knn_neighbors=DEFAULT_KNN_NEIGHBORS,
               random_seed=RANDOM_SEED, use_cache=True, verbose=True):
    """Run nested K-fold CV on any requested subset of the supported models."""
    model_names = normalize_model_names(model_names or ["all"])
    X, y, label_to_name, image_paths = load_dataset(
        dataset_path,
        image_size=image_size,
        max_images=max_images,
        max_images_per_person=max_images_per_person,
        cache_name="part_d",
        use_cache=use_cache,
        random_seed=random_seed,
        verbose=verbose,
    )

    cv_stats = validate_nested_cv_setup(y, outer_folds=outer_folds, inner_folds=inner_folds)
    search_space = build_search_space(
        model_names,
        max_supported_components=min(len(X), X.shape[1]),
        component_grid=component_grid,
        knn_neighbors=knn_neighbors,
    )

    print("=" * 70)
    print("PART D - NESTED K-FOLD MODEL SELECTION")
    print("=" * 70)
    print(f"Samples loaded: {len(X)}")
    print(f"Classes: {len(label_to_name)}")
    print(f"Models evaluated: {', '.join(MODEL_SPECS[name].display_name for name in model_names)}")
    print(f"Outer folds: {outer_folds}")
    print(f"Inner folds: {inner_folds}")
    print(f"Smallest class size: {cv_stats['min_class_count']}")

    nested_results = run_nested_k_fold_cv(
        X,
        y,
        label_to_name,
        search_space,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=random_seed,
        verbose=verbose,
    )

    print("\nOuter-fold summary:")
    summary_rows = summarize_nested_cv_results(nested_results)
    for row in summary_rows:
        print(f"\n{row['display_name']}")
        print(f"  Mean outer accuracy: {row['average_outer_score']:.4f}")
        print(f"  Outer-fold std:      {row['std_outer_score']:.4f}")
        print(f"  Mean best inner acc: {row['mean_selected_inner_score']:.4f}")
        print(f"  Selected params:     {format_hyperparam_counter(row['hyperparam_counter'])}")

    winner = choose_best_model(summary_rows)
    print("\nSelected model family:")
    print(f"  Winner: {winner['display_name']}")
    print(f"  Mean outer accuracy: {winner['average_outer_score']:.4f}")

    final_selection = select_hyperparameters_with_inner_cv(
        X,
        y,
        num_classes=len(label_to_name),
        model_name=winner["model_name"],
        hyperparameter_grid=search_space[winner["model_name"]],
        inner_folds=inner_folds,
        seed=random_seed + 1000,
    )
    final_model = fit_final_model(
        X,
        y,
        label_to_name,
        model_name=winner["model_name"],
        hyperparams=final_selection["best_hyperparams"],
        image_size=image_size,
    )

    print("\nRetrained final model:")
    print(f"  Model: {final_model.display_name}")
    print(f"  Hyperparameters: {format_hyperparams(final_selection['best_hyperparams'])}")
    print(f"  Mean inner accuracy on full labeled set: {final_selection['best_score']:.4f}")

    if image_paths:
        predicted_label, scores = predict_single_image_with_model(final_model, image_paths[0])
        true_label = label_to_name[y[0]]
        print("\nSample prediction:")
        print(f"  Image: {image_paths[0]}")
        print(f"  True label: {true_label}")
        print(f"  Predicted label: {predicted_label}")
        if scores is not None:
            print(f"  Top score: {scores.max():.4f}")

    return {
        "nested_results": nested_results,
        "summary_rows": summary_rows,
        "winner": winner,
        "final_selection": final_selection,
        "final_model": final_model,
    }


def build_parser():
    """Build the Part D CLI."""
    parser = argparse.ArgumentParser(description="Run Part D nested K-fold model selection.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the labeled dataset zip.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names to evaluate: linear_least_squares nearest_class_mean knn all",
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
    parser.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS, help="Outer K in nested CV.")
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS, help="Inner K in nested CV.")
    parser.add_argument(
        "--component-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_COMPONENT_GRID),
        help="PCA component counts to try.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        nargs="+",
        default=list(DEFAULT_KNN_NEIGHBORS),
        help="Neighbor counts to try for k-NN.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset caching.")
    parser.add_argument("--quiet", action="store_true", help="Reduce loader output.")
    return parser


def main(argv=None):
    """CLI entry point for Part D."""
    args = build_parser().parse_args(argv)
    image_size = parse_image_size(args.image_height, args.image_width)
    run_part_d(
        dataset_path=args.dataset,
        model_names=args.models,
        image_size=image_size,
        max_images=args.max_images,
        max_images_per_person=parse_max_images_per_person(args.max_images_per_person),
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        component_grid=args.component_grid,
        knn_neighbors=args.knn_neighbors,
        random_seed=args.seed,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )
    return 0

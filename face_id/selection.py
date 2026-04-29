"""Hyperparameter tuning and nested cross-validation utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .data import load_pgm, preprocess_image
from .metrics import accuracy_score
from .model_registry import (
    MODEL_SPECS,
    fit_projected_model,
    predict_projected_model,
    restrict_search_space,
)
from .pca import fit_pca, transform_pca
from .splits import stratified_k_fold_indices


@dataclass
class TrainedModel:
    """Fully trained classical model including preprocessing metadata."""

    model_name: str
    hyperparams: dict
    num_classes: int
    label_to_name: dict
    image_size: tuple | None
    pca_state: object
    projected_model: dict

    @property
    def display_name(self):
        return MODEL_SPECS[self.model_name].display_name


def format_hyperparams(hyperparams):
    """Render a hyperparameter dictionary in stable key order."""
    return ", ".join(
        f"{key}={value}"
        for key, value in sorted(hyperparams.items())
    )


def format_hyperparam_counter(counter):
    """Format selected-hyperparameter frequencies."""
    if not counter:
        return "n/a"
    return "; ".join(
        f"{hyperparams} ({count}x)"
        for hyperparams, count in counter.items()
    )


def hyperparams_to_key(hyperparams):
    """Create a hashable key for one hyperparameter setting."""
    return tuple(sorted(hyperparams.items()))


def _max_components_from_grid(hyperparameter_grid):
    return max(int(hyperparams["n_components"]) for hyperparams in hyperparameter_grid)


def select_hyperparameters_with_validation_split(X_train, y_train, X_val, y_val,
                                                 num_classes, model_name,
                                                 hyperparameter_grid):
    """
    Tune one model family using a single validation split.

    Returns:
        dict: best hyperparameters, best validation score, and all tried settings.
    """
    model_spec = MODEL_SPECS[model_name]
    max_valid_components = min(len(X_train), X_train.shape[1])
    valid_grid = restrict_search_space(model_name, hyperparameter_grid, max_valid_components)

    if model_spec.uses_pca:
        max_requested_components = _max_components_from_grid(valid_grid)
        pca_state, train_features = fit_pca(X_train, max_requested_components)
        val_features = transform_pca(X_val, pca_state)
    else:
        train_features = X_train
        val_features = X_val

    all_scores = []
    best_result = None

    for hyperparams in valid_grid:
        if model_spec.uses_pca:
            n_components = int(hyperparams["n_components"])
            selected_train_features = train_features[:, :n_components]
            selected_val_features = val_features[:, :n_components]
        else:
            selected_train_features = train_features
            selected_val_features = val_features

        projected_model = fit_projected_model(
            model_name,
            selected_train_features,
            y_train,
            num_classes,
            hyperparams,
        )
        y_train_pred, _ = predict_projected_model(projected_model, selected_train_features)
        y_val_pred, _ = predict_projected_model(projected_model, selected_val_features)

        result = {
            "hyperparams": dict(hyperparams),
            "train_score": accuracy_score(y_train, y_train_pred),
            "validation_score": accuracy_score(y_val, y_val_pred),
        }
        all_scores.append(result)

        if (
            best_result is None or
            result["validation_score"] > best_result["validation_score"]
        ):
            best_result = result

    return {
        "best_hyperparams": dict(best_result["hyperparams"]),
        "best_validation_score": float(best_result["validation_score"]),
        "best_train_score": float(best_result["train_score"]),
        "all_scores": all_scores,
    }


def select_hyperparameters_with_inner_cv(X, y, num_classes, model_name,
                                         hyperparameter_grid, inner_folds, seed=42):
    """
    Tune one model family using inner-loop stratified K-fold cross-validation.

    Returns:
        dict: best hyperparameters plus the mean inner validation score.
    """
    model_spec = MODEL_SPECS[model_name]
    inner_splits = stratified_k_fold_indices(y, inner_folds, seed=seed)
    max_valid_components = min(
        min(len(train_indices), X.shape[1])
        for train_indices, _ in inner_splits
    )
    valid_grid = restrict_search_space(model_name, hyperparameter_grid, max_valid_components)
    scores_by_key = {
        hyperparams_to_key(hyperparams): []
        for hyperparams in valid_grid
    }
    max_requested_components = (
        _max_components_from_grid(valid_grid)
        if model_spec.uses_pca else
        None
    )

    for inner_train_idx, inner_val_idx in inner_splits:
        X_inner_train = X[inner_train_idx]
        y_inner_train = y[inner_train_idx]
        X_inner_val = X[inner_val_idx]
        y_inner_val = y[inner_val_idx]

        if model_spec.uses_pca:
            inner_max_components = min(
                max_requested_components,
                len(X_inner_train),
                X_inner_train.shape[1],
            )
            pca_state, train_features = fit_pca(X_inner_train, inner_max_components)
            val_features = transform_pca(X_inner_val, pca_state)
        else:
            train_features = X_inner_train
            val_features = X_inner_val

        for hyperparams in valid_grid:
            if model_spec.uses_pca:
                n_components = int(hyperparams["n_components"])
                selected_train_features = train_features[:, :n_components]
                selected_val_features = val_features[:, :n_components]
            else:
                selected_train_features = train_features
                selected_val_features = val_features

            projected_model = fit_projected_model(
                model_name,
                selected_train_features,
                y_inner_train,
                num_classes,
                hyperparams,
            )
            y_inner_val_pred, _ = predict_projected_model(
                projected_model,
                selected_val_features,
            )
            scores_by_key[hyperparams_to_key(hyperparams)].append(
                accuracy_score(y_inner_val, y_inner_val_pred)
            )

    all_scores = []
    best_result = None

    for hyperparams in valid_grid:
        fold_scores = scores_by_key[hyperparams_to_key(hyperparams)]
        result = {
            "hyperparams": dict(hyperparams),
            "fold_scores": list(fold_scores),
            "mean_score": float(np.mean(fold_scores)),
        }
        all_scores.append(result)
        if best_result is None or result["mean_score"] > best_result["mean_score"]:
            best_result = result

    return {
        "best_hyperparams": dict(best_result["hyperparams"]),
        "best_score": float(best_result["mean_score"]),
        "all_scores": all_scores,
    }


def fit_final_model(X_train, y_train, label_to_name, model_name, hyperparams,
                    image_size):
    """Retrain a selected model on the full available labeled set."""
    num_classes = len(label_to_name)
    model_spec = MODEL_SPECS[model_name]

    if model_spec.uses_pca:
        n_components = min(int(hyperparams["n_components"]), len(X_train), X_train.shape[1])
        pca_state, train_features = fit_pca(X_train, n_components)
        selected_train_features = train_features[:, :n_components]
    else:
        pca_state = None
        selected_train_features = X_train

    projected_model = fit_projected_model(
        model_name,
        selected_train_features,
        y_train,
        num_classes,
        hyperparams,
    )
    return TrainedModel(
        model_name=model_name,
        hyperparams=dict(hyperparams),
        num_classes=num_classes,
        label_to_name=dict(label_to_name),
        image_size=image_size,
        pca_state=pca_state,
        projected_model=projected_model,
    )


def predict_trained_model(trained_model, X):
    """Predict labels for raw flattened images using a trained model."""
    if trained_model.pca_state is None:
        features = X
    else:
        features = transform_pca(X, trained_model.pca_state)
    return predict_projected_model(trained_model.projected_model, features)


def predict_single_image_with_model(trained_model, image_path):
    """Predict one image from disk or from inside the dataset zip."""
    image = preprocess_image(load_pgm(image_path), target_size=trained_model.image_size)
    y_pred, scores = predict_trained_model(trained_model, image.flatten()[np.newaxis, :])
    predicted_index = int(y_pred[0])
    return trained_model.label_to_name[predicted_index], None if scores is None else scores[0]


def run_nested_k_fold_cv(X, y, label_to_name, model_search_space, outer_folds,
                         inner_folds, seed=42, verbose=True):
    """Run nested CV for any subset of the supported model families."""
    num_classes = len(label_to_name)
    outer_splits = stratified_k_fold_indices(y, outer_folds, seed=seed)
    results = {
        model_name: {
            "outer_scores": [],
            "selected_hyperparams": [],
            "selected_inner_scores": [],
            "outer_fold_details": [],
        }
        for model_name in model_search_space
    }

    for outer_fold_index, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
        X_outer_train = X[outer_train_idx]
        y_outer_train = y[outer_train_idx]
        X_outer_test = X[outer_test_idx]
        y_outer_test = y[outer_test_idx]

        if verbose:
            print(f"\n  Outer fold {outer_fold_index}/{outer_folds}")
            print(f"    Outer train size: {len(outer_train_idx)}")
            print(f"    Outer test size:  {len(outer_test_idx)}")

        chosen_settings = {}
        for model_name, hyperparameter_grid in model_search_space.items():
            selection = select_hyperparameters_with_inner_cv(
                X_outer_train,
                y_outer_train,
                num_classes,
                model_name,
                hyperparameter_grid,
                inner_folds=inner_folds,
                seed=seed + outer_fold_index,
            )
            chosen_settings[model_name] = selection
            if verbose:
                print(
                    f"    {MODEL_SPECS[model_name].display_name}: "
                    f"best inner acc={selection['best_score']:.4f} "
                    f"with {format_hyperparams(selection['best_hyperparams'])}"
                )

        pca_features = {}
        pca_model_names = [
            model_name
            for model_name in chosen_settings
            if MODEL_SPECS[model_name].uses_pca
        ]

        if pca_model_names:
            outer_max_components = max(
                int(chosen_settings[model_name]["best_hyperparams"]["n_components"])
                for model_name in pca_model_names
            )
            outer_max_components = min(
                outer_max_components,
                len(X_outer_train),
                X_outer_train.shape[1],
            )
            pca_state, train_features = fit_pca(X_outer_train, outer_max_components)
            test_features = transform_pca(X_outer_test, pca_state)
            pca_features["train"] = train_features
            pca_features["test"] = test_features

        for model_name, selection in chosen_settings.items():
            hyperparams = selection["best_hyperparams"]
            if MODEL_SPECS[model_name].uses_pca:
                n_components = int(hyperparams["n_components"])
                selected_train_features = pca_features["train"][:, :n_components]
                selected_test_features = pca_features["test"][:, :n_components]
            else:
                selected_train_features = X_outer_train
                selected_test_features = X_outer_test

            projected_model = fit_projected_model(
                model_name,
                selected_train_features,
                y_outer_train,
                num_classes,
                hyperparams,
            )
            y_outer_pred, _ = predict_projected_model(
                projected_model,
                selected_test_features,
            )
            outer_score = accuracy_score(y_outer_test, y_outer_pred)

            results[model_name]["outer_scores"].append(float(outer_score))
            results[model_name]["selected_hyperparams"].append(dict(hyperparams))
            results[model_name]["selected_inner_scores"].append(float(selection["best_score"]))
            results[model_name]["outer_fold_details"].append({
                "outer_fold": outer_fold_index,
                "outer_score": float(outer_score),
                "inner_score": float(selection["best_score"]),
                "hyperparams": dict(hyperparams),
            })

            if verbose:
                print(
                    f"    {MODEL_SPECS[model_name].display_name}: "
                    f"outer test acc={outer_score:.4f}"
                )

    return results


def summarize_nested_cv_results(results):
    """Summarize mean outer performance, variability, and chosen hyperparameters."""
    rows = []
    for model_name, model_results in results.items():
        outer_scores = np.asarray(model_results["outer_scores"], dtype=np.float32)
        rows.append({
            "model_name": model_name,
            "display_name": MODEL_SPECS[model_name].display_name,
            "average_outer_score": float(outer_scores.mean()),
            "std_outer_score": float(outer_scores.std(ddof=1)) if len(outer_scores) > 1 else 0.0,
            "mean_selected_inner_score": float(np.mean(model_results["selected_inner_scores"])),
            "hyperparam_counter": Counter(
                format_hyperparams(hyperparams)
                for hyperparams in model_results["selected_hyperparams"]
            ),
        })
    return rows


def choose_best_model(summary_rows):
    """Pick the model with the best mean outer-fold score."""
    return max(
        summary_rows,
        key=lambda row: (row["average_outer_score"], -row["std_outer_score"]),
    )

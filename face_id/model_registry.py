"""Model registry and hyperparameter-grid helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_COMPONENT_GRID, DEFAULT_KNN_NEIGHBORS
from .models import knn, linear_least_squares, nearest_class_mean, svm


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one supported model family."""

    name: str
    display_name: str
    uses_pca: bool
    fit_fn: object
    predict_fn: object


MODEL_SPECS = {
    "linear_least_squares": ModelSpec(
        name="linear_least_squares",
        display_name="Linear least squares",
        uses_pca=True,
        fit_fn=linear_least_squares.fit,
        predict_fn=linear_least_squares.predict,
    ),
    "nearest_class_mean": ModelSpec(
        name="nearest_class_mean",
        display_name="Nearest class mean",
        uses_pca=True,
        fit_fn=nearest_class_mean.fit,
        predict_fn=nearest_class_mean.predict,
    ),
    "svm": ModelSpec(
        name="svm",
        display_name="Support Vector Machine",
        uses_pca=True,
        fit_fn=svm.fit,
        predict_fn=svm.predict,
    ),
    "knn": ModelSpec(
        name="knn",
        display_name="k-NN",
        uses_pca=False,
        fit_fn=knn.fit,
        predict_fn=knn.predict,
    ),
}


def normalize_model_names(model_names):
    """Resolve 'all' and validate requested model names."""
    if not model_names:
        return list(MODEL_SPECS)

    requested = [name.strip() for name in model_names]
    if "all" in requested:
        return list(MODEL_SPECS)

    unknown = [name for name in requested if name not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown models requested: {', '.join(sorted(unknown))}")

    unique_names = []
    for name in requested:
        if name not in unique_names:
            unique_names.append(name)
    return unique_names


def available_model_names():
    """Return all supported model names."""
    return tuple(MODEL_SPECS)


def build_search_space(model_names, max_supported_components, component_grid=None,
                       knn_neighbors=None):
    """Build the hyperparameter grid for the requested model families."""
    from .config import DEFAULT_SVM_REG_GRID, DEFAULT_SVM_EPOCHS_GRID, DEFAULT_SVM_LEARNING_RATE_GRID
    
    model_names = normalize_model_names(model_names)
    component_grid = DEFAULT_COMPONENT_GRID if component_grid is None else component_grid
    knn_neighbors = DEFAULT_KNN_NEIGHBORS if knn_neighbors is None else knn_neighbors

    component_values = sorted({
        int(value)
        for value in component_grid
        if 1 <= int(value) <= int(max_supported_components)
    })
    if not component_values:
        component_values = [max(1, int(max_supported_components))]

    neighbor_values = sorted({max(1, int(value)) for value in knn_neighbors})
    search_space = {}

    for model_name in model_names:
        if model_name == "knn":
            search_space[model_name] = [
                {"n_components": n_components, "k": k}
                for n_components in component_values
                for k in neighbor_values
            ]
        elif model_name == "svm":
            search_space[model_name] = [
                {
                    "n_components": n_components,
                    "reg_strength": reg,
                    "epochs": epochs,
                    "learning_rate": lr
                }
                for n_components in component_values
                for reg in DEFAULT_SVM_REG_GRID
                for epochs in DEFAULT_SVM_EPOCHS_GRID
                for lr in DEFAULT_SVM_LEARNING_RATE_GRID
            ]
        else:
            search_space[model_name] = [
                {"n_components": n_components}
                for n_components in component_values
            ]

    return search_space


def restrict_search_space(model_name, hyperparameter_grid, max_n_components):
    """Keep only valid PCA dimensions, with a small safe fallback if needed."""
    max_n_components = max(1, int(max_n_components))
    valid_grid = [
        dict(hyperparams)
        for hyperparams in hyperparameter_grid
        if int(hyperparams["n_components"]) <= max_n_components
    ]
    if valid_grid:
        return valid_grid

    if model_name == "knn":
        neighbor_values = sorted({
            int(hyperparams["k"])
            for hyperparams in hyperparameter_grid
            if "k" in hyperparams
        }) or [1]
        return [
            {"n_components": max_n_components, "k": k}
            for k in neighbor_values
        ]

    return [{"n_components": max_n_components}]


def fit_projected_model(model_name, Z_train, y_train, num_classes, hyperparams):
    """Fit one model family on already-projected PCA features."""
    model_state = MODEL_SPECS[model_name].fit_fn(
        Z_train,
        y_train,
        num_classes,
        hyperparams,
    )
    model_state.update({
        "model_name": model_name,
        "hyperparams": dict(hyperparams),
        "num_classes": int(num_classes),
    })
    return model_state


def predict_projected_model(model_state, Z):
    """Predict with one projected model."""
    model_name = model_state["model_name"]
    return MODEL_SPECS[model_name].predict_fn(model_state, Z)

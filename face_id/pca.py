"""PCA helpers used by the classical models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAState:
    """Fitted PCA projection state."""

    mean_face: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray


def fit_pca(X_train, n_components):
    """Fit PCA using NumPy SVD and return projected training features."""
    n_components = max(1, min(int(n_components), X_train.shape[0], X_train.shape[1]))
    mean_face = X_train.mean(axis=0)
    centered = X_train - mean_face
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    explained_variance = (singular_values[:n_components] ** 2) / max(X_train.shape[0] - 1, 1)
    Z_train = centered @ components.T
    return PCAState(mean_face, components, explained_variance), Z_train


def transform_pca(X, pca_state):
    """Project raw feature vectors using a fitted PCA state."""
    centered = X - pca_state.mean_face
    return centered @ pca_state.components.T

"""PCA helpers used by the classical models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .config import CACHE_DIR


PCA_CACHE_VERSION = 1
PCA_CACHE_NAME = "pca_basis"


@dataclass
class PCAState:
    """Fitted PCA projection state."""

    mean_face: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray


def _normalize_n_components(X_train, n_components):
    """Clamp the requested PCA dimension to the valid SVD rank."""
    n_components = max(1, min(int(n_components), X_train.shape[0], X_train.shape[1]))
    return n_components


def _pca_cache_key(X_train, n_components):
    """Build a stable cache key for one exact PCA fit request."""
    contiguous = np.ascontiguousarray(X_train)
    data_digest = hashlib.sha1(contiguous.tobytes()).hexdigest()
    return (
        f"v{PCA_CACHE_VERSION}|shape={contiguous.shape}|dtype={contiguous.dtype}|"
        f"n_components={n_components}|data={data_digest}"
    )


def _pca_cache_path(cache_key):
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"{PCA_CACHE_NAME}_{digest}.npz"


def _load_cached_pca(cache_path):
    with np.load(cache_path, allow_pickle=False) as data:
        pca_state = PCAState(
            mean_face=data["mean_face"],
            components=data["components"],
            explained_variance=data["explained_variance"],
        )
        return pca_state, data["Z_train"]


def _save_cached_pca(cache_path, pca_state, Z_train):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        mean_face=pca_state.mean_face,
        components=pca_state.components,
        explained_variance=pca_state.explained_variance,
        Z_train=Z_train,
    )


def fit_pca(X_train, n_components, use_cache=True):
    """Fit PCA using NumPy SVD and return projected training features."""
    n_components = _normalize_n_components(X_train, n_components)

    if use_cache:
        cache_path = _pca_cache_path(_pca_cache_key(X_train, n_components))
        if cache_path.exists():
            try:
                return _load_cached_pca(cache_path)
            except (KeyError, OSError, ValueError):
                pass

    mean_face = X_train.mean(axis=0)
    centered = X_train - mean_face
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    explained_variance = (singular_values[:n_components] ** 2) / max(X_train.shape[0] - 1, 1)
    Z_train = centered @ components.T
    pca_state = PCAState(mean_face, components, explained_variance)

    if use_cache:
        _save_cached_pca(cache_path, pca_state, Z_train)

    return pca_state, Z_train


def transform_pca(X, pca_state):
    """Project raw feature vectors using a fitted PCA state."""
    centered = X - pca_state.mean_face
    return centered @ pca_state.components.T

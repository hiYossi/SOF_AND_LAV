"""Nearest class mean classifier in PCA space."""

import numpy as np


def fit(Z_train, y_train, num_classes, hyperparams):
    """Fit one centroid per class."""
    del hyperparams
    Z_train = np.asarray(Z_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    num_classes = int(num_classes)

    means = np.zeros((num_classes, Z_train.shape[1]), dtype=np.float32)
    for class_index in range(num_classes):
        class_vectors = Z_train[y_train == class_index]
        if len(class_vectors) > 0:
            means[class_index] = class_vectors.mean(axis=0)

    return {"means": means}


def predict(model_state, Z):
    """Predict the label of the nearest class centroid."""
    Z = np.asarray(Z, dtype=np.float32)
    means = model_state["means"]

    sample_sq_norms = np.sum(Z * Z, axis=1, keepdims=True)
    mean_sq_norms = np.sum(means * means, axis=1)
    distances = sample_sq_norms + mean_sq_norms[np.newaxis, :] - 2.0 * (Z @ means.T)
    distances = np.maximum(distances, 0.0)

    scores = -distances
    return np.argmin(distances, axis=1), scores

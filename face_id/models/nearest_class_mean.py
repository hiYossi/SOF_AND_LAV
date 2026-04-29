"""Nearest-class-mean classifier in PCA space."""

import numpy as np


def fit(Z_train, y_train, num_classes, hyperparams):
    """Compute one mean vector per class."""
    del hyperparams
    class_means = np.zeros((num_classes, Z_train.shape[1]), dtype=np.float32)
    for class_index in range(num_classes):
        mask = y_train == class_index
        if np.any(mask):
            class_means[class_index] = Z_train[mask].mean(axis=0)
    return {"class_means": class_means}


def predict(model_state, Z):
    """Predict labels using Euclidean distance to the class means."""
    distances = np.linalg.norm(
        Z[:, np.newaxis, :] - model_state["class_means"][np.newaxis, :, :],
        axis=2,
    )
    return np.argmin(distances, axis=1), -distances

"""Multiclass linear least-squares classifier."""

import numpy as np


def _one_hot_encode(y, num_classes):
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def fit(Z_train, y_train, num_classes, hyperparams):
    """Fit a least-squares linear classifier in PCA space."""
    del hyperparams
    Z1 = np.hstack([Z_train, np.ones((len(Z_train), 1), dtype=Z_train.dtype)])
    Y = _one_hot_encode(y_train, num_classes)
    W = np.linalg.pinv(Z1) @ Y
    return {"W": W}


def predict(model_state, Z):
    """Predict labels and raw class scores."""
    Z1 = np.hstack([Z, np.ones((len(Z), 1), dtype=Z.dtype)])
    scores = Z1 @ model_state["W"]
    return np.argmax(scores, axis=1), scores

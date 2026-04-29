"""Linear one-vs-rest SVM trained with mini-batch SGD."""

from __future__ import annotations

import numpy as np


DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.2
DEFAULT_RANDOM_SEED = 42


def _augment_bias(features):
    """Append a bias column to the feature matrix."""
    return np.hstack([
        features.astype(np.float32, copy=False),
        np.ones((len(features), 1), dtype=np.float32),
    ])


def _one_vs_rest_targets(y, num_classes):
    """Encode one-vs-rest targets as {-1, +1}."""
    targets = -np.ones((len(y), num_classes), dtype=np.float32)
    targets[np.arange(len(y)), y] = 1.0
    return targets


def fit(Z_train, y_train, num_classes, hyperparams):
    """
    Fit a linear one-vs-rest SVM with hinge loss and L2 regularization.

    Hyperparameters:
        reg_strength: L2 regularization multiplier.
        epochs: number of passes over the training data.
        batch_size: SGD mini-batch size.
        learning_rate: base learning rate.
        seed: RNG seed for shuffling.
    """
    reg_strength = float(hyperparams.get("reg_strength", 1e-3))
    epochs = int(hyperparams.get("epochs", DEFAULT_EPOCHS))
    batch_size = int(hyperparams.get("batch_size", DEFAULT_BATCH_SIZE))
    learning_rate = float(hyperparams.get("learning_rate", DEFAULT_LEARNING_RATE))
    seed = int(hyperparams.get("seed", DEFAULT_RANDOM_SEED))

    X = _augment_bias(Z_train)
    Y = _one_vs_rest_targets(y_train, num_classes)
    weights = np.zeros((X.shape[1], num_classes), dtype=np.float32)
    regularization_mask = np.ones_like(weights, dtype=np.float32)
    regularization_mask[-1, :] = 0.0

    rng = np.random.default_rng(seed)
    
    # Adding Momentum for faster convergence
    velocity = np.zeros_like(weights, dtype=np.float32)
    momentum_factor = 0.9

    for epoch in range(epochs):
        permutation = rng.permutation(len(X))
        # Decay learning rate slightly
        step_size = learning_rate / (1.0 + 0.1 * epoch)

        for start in range(0, len(X), batch_size):
            batch_indices = permutation[start:start + batch_size]
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]

            margins = Y_batch * (X_batch @ weights)
            active = (margins < 1.0).astype(np.float32)

            hinge_gradient = -(X_batch.T @ (Y_batch * active)) / max(len(X_batch), 1)
            regularization_gradient = reg_strength * (weights * regularization_mask)
            gradient = regularization_gradient + hinge_gradient
            
            # Momentum update rule
            velocity = momentum_factor * velocity - step_size * gradient
            weights += velocity

    return {"W": weights}


def predict(model_state, Z):
    """Predict labels and raw decision scores."""
    X = _augment_bias(Z)
    scores = X @ model_state["W"]
    return np.argmax(scores, axis=1), scores

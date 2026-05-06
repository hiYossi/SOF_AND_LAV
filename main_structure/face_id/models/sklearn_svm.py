"""Linear multi-class SVM using sklearn.svm.LinearSVC.

Wraps sklearn's optimised LibLinear-based solver as a drop-in replacement
for the hand-rolled SGD SVM.  The interface mirrors all other models in
this package: fit() returns a plain dict, predict() returns (labels, scores).
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC

DEFAULT_C        = 1.0
DEFAULT_MAX_ITER = 2000


def fit(Z_train, y_train, num_classes, hyperparams):
    """Fit a one-vs-rest LinearSVC on PCA-projected features.

    Hyperparameters:
        C        : inverse regularisation strength (higher = less regularised).
        max_iter : solver iteration cap.
    """
    C        = float(hyperparams.get("C",        DEFAULT_C))
    max_iter = int(hyperparams.get("max_iter",   DEFAULT_MAX_ITER))

    clf = LinearSVC(
        C=C,
        max_iter=max_iter,
        multi_class="ovr",
        random_state=42,
        dual="auto",
    )
    clf.fit(Z_train, y_train)
    return {"clf": clf}


def predict(model_state, Z):
    """Predict labels and per-class decision scores."""
    clf    = model_state["clf"]
    labels = clf.predict(Z)
    scores = clf.decision_function(Z)          # (n_samples, n_classes)
    return labels, scores

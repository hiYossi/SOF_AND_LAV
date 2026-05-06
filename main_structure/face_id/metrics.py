"""Evaluation metrics."""

import numpy as np


def accuracy_score(y_true, y_pred):
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes):
    """Compute a dense confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix

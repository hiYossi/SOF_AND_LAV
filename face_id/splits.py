"""Stratified dataset splitting utilities."""

import numpy as np


def stratified_holdout_indices(y, holdout_ratio, seed=42):
    """Create one stratified holdout split."""
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices = []
    holdout_indices = []

    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        class_indices = rng.permutation(class_indices)
        holdout_count = int(round(len(class_indices) * holdout_ratio))
        if len(class_indices) > 1:
            holdout_count = max(1, min(holdout_count, len(class_indices) - 1))
        else:
            holdout_count = 0

        holdout_indices.extend(class_indices[:holdout_count].tolist())
        train_indices.extend(class_indices[holdout_count:].tolist())

    train_indices = np.asarray(train_indices, dtype=np.int32)
    holdout_indices = np.asarray(holdout_indices, dtype=np.int32)
    rng.shuffle(train_indices)
    rng.shuffle(holdout_indices)
    return train_indices, holdout_indices


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Return stratified train/test arrays."""
    train_indices, test_indices = stratified_holdout_indices(y, test_ratio, seed=seed)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def stratified_k_fold_indices(y, n_splits, seed=42):
    """Build stratified K-fold train/test index pairs."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    unique_labels, counts = np.unique(y, return_counts=True)
    if np.any(counts < n_splits):
        raise ValueError(
            f"Cannot create {n_splits}-fold stratified CV because the smallest class "
            f"has only {int(counts.min())} samples."
        )

    rng = np.random.default_rng(seed)
    fold_test_parts = [[] for _ in range(n_splits)]

    for class_label in unique_labels:
        class_indices = np.where(y == class_label)[0]
        class_indices = rng.permutation(class_indices)
        for fold_index, chunk in enumerate(np.array_split(class_indices, n_splits)):
            fold_test_parts[fold_index].extend(chunk.tolist())

    all_indices = np.arange(len(y), dtype=np.int32)
    folds = []

    for fold_index in range(n_splits):
        test_indices = np.asarray(fold_test_parts[fold_index], dtype=np.int32)
        rng.shuffle(test_indices)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        rng.shuffle(train_indices)
        folds.append((train_indices, test_indices))

    return folds


def validate_nested_cv_setup(y, outer_folds, inner_folds):
    """Validate that each class has enough samples for nested CV."""
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2.")

    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())
    if min_class_count < outer_folds:
        raise ValueError(
            f"Need at least {outer_folds} samples per class for the outer loop, "
            f"but the smallest class has only {min_class_count}."
        )

    min_outer_train_count = min(
        int(count - np.ceil(count / outer_folds))
        for count in counts
    )
    if min_outer_train_count < inner_folds:
        raise ValueError(
            f"Need at least {inner_folds} samples per class inside each outer training split, "
            f"but the worst-case outer training count is only {min_outer_train_count}."
        )

    return {
        "min_class_count": min_class_count,
        "max_class_count": int(counts.max()),
        "min_outer_train_count": int(min_outer_train_count),
    }

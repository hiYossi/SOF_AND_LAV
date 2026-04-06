"""k-nearest-neighbors classifier in PCA space."""

import numpy as np

from ..config import DEFAULT_KNN_BATCH_SIZE


def fit(Z_train, y_train, num_classes, hyperparams):
    """Store training vectors for lazy k-NN prediction."""
    return {
        "Z_train": np.asarray(Z_train, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=np.int32),
        "num_classes": int(num_classes),
        "k": int(hyperparams["k"]),
        "batch_size": DEFAULT_KNN_BATCH_SIZE,
    }


def predict(model_state, Z):
    """Predict labels using batched Euclidean-distance search."""
    Z_train = model_state["Z_train"]
    y_train = model_state["y_train"]
    num_classes = model_state["num_classes"]
    k = max(1, min(int(model_state["k"]), len(Z_train)))
    batch_size = max(1, int(model_state.get("batch_size", DEFAULT_KNN_BATCH_SIZE)))

    y_pred = np.empty(len(Z), dtype=y_train.dtype)
    vote_scores = np.zeros((len(Z), num_classes), dtype=np.float32)
    train_sq_norms = np.sum(Z_train * Z_train, axis=1)

    for start in range(0, len(Z), batch_size):
        stop = min(start + batch_size, len(Z))
        Z_batch = Z[start:stop]
        batch_sq_norms = np.sum(Z_batch * Z_batch, axis=1, keepdims=True)
        distances = batch_sq_norms + train_sq_norms[np.newaxis, :] - 2.0 * (Z_batch @ Z_train.T)
        distances = np.maximum(distances, 0.0)

        nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_labels = y_train[nearest_indices]

        for row_index, labels in enumerate(nearest_labels):
            counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
            vote_scores[start + row_index] = counts / float(k)
            y_pred[start + row_index] = np.argmax(counts)

    return y_pred, vote_scores

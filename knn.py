import numpy as np
from pathlib import Path

import linear_model


DATASET_PATH = linear_model.DATASET_PATH
CACHE_PATH = Path(__file__).with_name("knn_cache.npz")
IMAGE_SIZE = (64, 64)
RANDOM_SEED = 42
TEST_RATIO = 0.2
MAX_IMAGES_PER_PERSON = 1000
K_NEIGHBORS = 7
BATCH_SIZE = 128
VERBOSE = True


def load_dataset(person_ids=None, max_images_per_person=MAX_IMAGES_PER_PERSON,
                 image_size=IMAGE_SIZE):
    """Load a KNN-ready subset using the shared dataset loader."""
    X, y, label_to_name, image_paths = linear_model.load_dataset(
        DATASET_PATH,
        image_size=image_size,
        max_images=None,
        max_images_per_person=max_images_per_person,
        cache_path=CACHE_PATH,
        verbose=VERBOSE,
        random_seed=RANDOM_SEED,
    )

    if person_ids is not None:
        person_ids = np.array(list(person_ids), dtype=np.int32)
        keep_mask = np.isin(y, person_ids)
        X = X[keep_mask]
        y = y[keep_mask]
        image_paths = [path for path, keep in zip(image_paths, keep_mask) if keep]
        label_to_name = {
            int(label): name
            for label, name in label_to_name.items()
            if int(label) in set(person_ids.tolist())
        }

    # Preserve the old KNN scaling range while reusing the shared loader.
    X = X * 2.0 - 1.0
    return X, y, label_to_name, image_paths


def train_test_split(X, y, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_index = int((1 - test_ratio) * len(X))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def predict(X_test, X_train, y_train, k=K_NEIGHBORS, batch_size=BATCH_SIZE):
    """Predict labels in batches using vectorized squared distances."""
    if len(X_train) == 0:
        raise ValueError("Training set is empty.")

    y_pred = np.empty(len(X_test), dtype=y_train.dtype)
    train_sq_norms = np.sum(X_train * X_train, axis=1)

    for start in range(0, len(X_test), batch_size):
        stop = min(start + batch_size, len(X_test))
        X_batch = X_test[start:stop]
        batch_sq_norms = np.sum(X_batch * X_batch, axis=1, keepdims=True)
        distances = batch_sq_norms + train_sq_norms[np.newaxis, :] - 2.0 * (X_batch @ X_train.T)
        distances = np.maximum(distances, 0.0)

        nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_labels = y_train[nearest_indices]

        for row_idx, labels in enumerate(nearest_labels):
            y_pred[start + row_idx] = np.bincount(labels).argmax()

    return y_pred


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    person_ids = range(28)
    X, y, label_to_name, _ = load_dataset(person_ids, max_images_per_person=MAX_IMAGES_PER_PERSON)

    print("Loaded classes:", list(label_to_name.values()))
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=TEST_RATIO)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    y_pred = predict(X_test, X_train, y_train, k=K_NEIGHBORS)
    acc = accuracy(y_test, y_pred)

    print("KNN accuracy:", acc)


if __name__ == "__main__":
    main()

from main import *
def load_dataset(person_ids, max_images_per_person=20):
    X = []
    y = []

    for p in person_ids:
        for i in range(max_images_per_person):
            try:
                file_path = get_file(p, i)
                vec = load_pgm_vector(file_path)
                X.append(vec)
                y.append(p)
            except:
                continue

    return np.array(X), np.array(y)


def train_test_split(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_index = int((1 - test_ratio) * len(X))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def compute_distances(x, X_train):
    return np.sqrt(np.sum((X_train - x) ** 2, axis=1))


def predict_one(x, X_train, y_train, k=3):
    distances = compute_distances(x, X_train)
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    return np.bincount(nearest_labels).argmax()


def predict(X_test, X_train, y_train, k=3):
    y_pred = []
    for x in X_test:
        y_pred.append(predict_one(x, X_train, y_train, k))
    return np.array(y_pred)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    person_ids = range(28)
    X, y = load_dataset(person_ids, max_images_per_person=300)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

    y_pred = predict(X_test, X_train, y_train, k=5)
    acc = accuracy(y_test, y_pred)

    print("KNN accuracy:", acc)
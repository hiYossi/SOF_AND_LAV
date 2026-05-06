import numpy as np
import matplotlib.pyplot as plt
from main import *
from linear_model import *


def kmeans_manual(data, k, max_iters=30, tol=1e-4):
    """
    Manual implementation of K-Means Clustering.
    """
    n_samples, n_features = data.shape

    # 1. Initialize centroids randomly from the data points
    rng = np.random.default_rng(42)
    random_indices = rng.choice(n_samples, k, replace=False)
    centroids = data[random_indices].copy()

    print(f"Starting K-Means with k={k}...")

    for i in range(max_iters):
        # 2. Assign step: Calculate Euclidean distance to all centroids
        # Using broadcasting for speed: (N, 1, 50) - (1, K, 50) -> (N, K, 50)
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Update step: New centroid is the mean of assigned points
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            points_in_cluster = data[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = points_in_cluster.mean(axis=0)
            else:
                # If a cluster is empty, re-initialize it to a random point
                new_centroids[j] = data[rng.choice(n_samples)]

        # Check for convergence
        center_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Iteration {i + 1:2d} | Center shift: {center_shift:.4f}")

        if center_shift < tol:
            print(f"Converged at iteration {i + 1}.")
            break

    return labels, centroids


def calculate_purity(y_true, cluster_labels):
    """
    Measures how 'pure' each cluster is relative to the true identities.
    """
    total_correct = 0
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0: continue

        # Find the most frequent true label in this cluster
        labels_in_cluster = y_true[indices]
        most_common = np.bincount(labels_in_cluster).argmax()
        total_correct += np.sum(labels_in_cluster == most_common)

    return total_correct / len(y_true)


def main_bonus():
    # 1. Load Data
    print("Loading data...")
    X, y, label_to_name, _ = load_dataset(DATASET_PATH, IMAGE_SIZE)

    # Flatten X if it's 4D (from CNN) or just ensure it's (N, D)
    X_flat = X.reshape(X.shape[0], -1)

    # 2. Dimensionality Reduction (PCA) - Essential for K-Means to work on images
    print("Reducing dimensions with PCA (50 components)...")
    mean_face, components, _, Z = fit_pca(X_flat, n_components=50)

    # 3. Run Manual K-Means
    num_identities = len(np.unique(y))
    predicted_clusters, _ = kmeans_manual(Z, k=num_identities)

    # 4. Evaluation
    purity_score = calculate_purity(y, predicted_clusters)

    print("\n" + "=" * 30)
    print(f"BONUS PART RESULTS")
    print(f"Number of Clusters: {num_identities}")
    print(f"Purity Score: {purity_score:.4f}")
    print(f"Random Guess Baseline: {1 / num_identities:.4f}")
    print("=" * 30)

    if purity_score > (1 / num_identities) * 2:
        print(
            "Analysis: The clustering is significantly better than random, meaning it discovered some facial structures.")
    else:
        print("Analysis: The clustering is close to random, likely dominated by lighting/noise.")


if __name__ == "__main__":
    main_bonus()
"""
Quick-start demo: Load subset of faces, train, evaluate (runs in ~1 minute).

Configuration optimized for speed over maximum accuracy.
"""

import numpy as np
from pathlib import Path
from linear_model import load_dataset


# ============================================================================
# FAST CONFIGURATION
# ============================================================================

DATASET_PATH = Path(__file__).with_name("Train Set (Labeled)-20260405T164823Z-3-001.zip")
CACHE_PATH = Path(__file__).with_name("quick_start_cache.npz")
IMAGE_SIZE = None                  # Preserve original image size
N_COMPONENTS = 50                  # Fewer components = faster
TEST_RATIO = 0.2
RANDOM_SEED = 42
MAX_IMAGES_PER_PERSON = 90         # Load at most 50 images per person
VERBOSE = True


# ============================================================================
# MINIMAL IMPLEMENTATION (copy-paste friendly)
# ============================================================================


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split data with shuffle."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def fit_pca(X, n_comp):
    """Fit PCA using SVD."""
    mean = X.mean(axis=0)
    X_c = X - mean
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    comp = Vt[:n_comp]
    Z = X_c @ comp.T
    return mean, comp, Z


def predict(Z_test, mean, comp, W):
    """Predict using trained model."""
    X_c = Z_test - (Z_test.mean(axis=0) if len(Z_test) > 1 else 0)  # Already centered, but for consistency
    Z = X_c @ comp.T if len(Z_test.shape) == 2 else Z_test  # Handle if already projected
    # Actually, Z_test should already be projected in this flow
    Z1 = np.hstack([Z, np.ones((Z.shape[0], 1))])
    scores = Z1 @ W
    return np.argmax(scores, axis=1)


def accuracy(y_true, y_pred):
    """Compute accuracy."""
    return (y_true == y_pred).mean()


def main():
    print("\n" + "="*60)
    print("QUICK START: Face Recognition (optimized for speed)")
    print("="*60 + "\n")
    
    # Load subset
    print("[1] Loading dataset...")
    X, y, names, paths = load_dataset(
        DATASET_PATH,
        IMAGE_SIZE,
        max_images=None,
        max_images_per_person=MAX_IMAGES_PER_PERSON,
        cache_path=CACHE_PATH,
        verbose=VERBOSE,
        random_seed=RANDOM_SEED,
        log_prefix="    ",
    )
    print(f"    Loaded {len(X)} images from {len(names)} people")
    if IMAGE_SIZE is None:
        print(f"    Preserved original image size, flattened to {X.shape[1]} pixels")
    else:
        print(f"    Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} ({IMAGE_SIZE[0]*IMAGE_SIZE[1]} pixels)")
    
    # Split
    print("\n[2] Splitting into train/test...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, TEST_RATIO, RANDOM_SEED)
    print(f"    Train: {len(X_tr)}, Test: {len(X_te)}")
    
    # PCA
    print(f"\n[3] Fitting PCA ({N_COMPONENTS} components)...")
    mean, comp, Z_tr = fit_pca(X_tr, N_COMPONENTS)
    Z_te = (X_te - mean) @ comp.T
    print(f"    Reduced dimension: {X_tr.shape[1]} -> {Z_tr.shape[1]}")
    
    # Train classifier
    print("\n[4] Training linear classifier...")
    n_class = len(names)
    Z1_tr = np.hstack([Z_tr, np.ones((len(Z_tr), 1))])
    Y = np.zeros((len(y_tr), n_class))
    Y[np.arange(len(y_tr)), y_tr] = 1
    W = np.linalg.pinv(Z1_tr) @ Y
    print(f"    Trained {n_class}-class classifier")
    
    # Evaluate
    print("\n[5] Evaluating...")
    Z1_te = np.hstack([Z_te, np.ones((len(Z_te), 1))])
    y_tr_pred = np.argmax(Z1_tr @ W, axis=1)
    y_te_pred = np.argmax(Z1_te @ W, axis=1)
    
    train_acc = accuracy(y_tr, y_tr_pred)
    test_acc = accuracy(y_te, y_te_pred)
    
    print(f"    Train accuracy: {train_acc:.1%}")
    print(f"    Test accuracy:  {test_acc:.1%}")
    
    # Sample prediction
    if len(X_te) > 0:
        print("\n[6] Sample predictions:")
        for i in range(min(3, len(X_te))):
            true_id = y_te[i]
            pred_id = y_te_pred[i]
            true_name = names[true_id]
            pred_name = names[pred_id]
            scores = (Z1_te @ W)[i]
            conf = scores.max()
            match = "OK" if true_id == pred_id else "X"
            print(f"    {match} True: {true_name:12} | Pred: {pred_name:12} | Conf: {conf:.2f}")
    
    print("\n" + "="*60)
    print(f"Done! Test accuracy: {test_acc:.1%}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

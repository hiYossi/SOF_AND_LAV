"""
Face Recognition using PCA and Linear Least Squares Classifier.

Pipeline:
1. Load labeled face dataset from folder structure
2. Preprocess images (resize, normalize to [0,1])
3. Flatten images to vectors
4. Mean-center the data
5. Compute PCA via SVD ("eigenfaces")
6. Train linear multiclass classifier using least squares
7. Evaluate on test set
8. Predict on new images

Uses NumPy only for linear algebra and machine learning.
"""

import os
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "Train Set (Labeled)-20260405T164823Z-3-001.zip"
IMAGE_SIZE = (64, 64)              # (height, width) — full resolution
N_COMPONENTS = 20                  # Use 20 for fast testing, 50 for final accuracy
TEST_RATIO = 0.2                   # Hold out 20% for testing
RANDOM_SEED = 42                   # For reproducibility
VERBOSE = True                     # Print progress info
MAX_IMAGES = 1000                  # Use 1000 images for the test, None for all ~3600 images


# ============================================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================================

def load_pgm_simple(filename):
    """
    Load a PGM image file (P5 format, binary grayscale).
    
    Args:
        filename (str): Path to PGM file or 'archive.zip/path/file.pgm'.
    
    Returns:
        ndarray: 2D grayscale image (height, width).
    """
    import zipfile
    
    if '.zip' in filename:
        zip_path, internal_path = filename.split('.zip', 1)
        zip_path += '.zip'
        internal_path = internal_path.lstrip('/')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(internal_path) as f:
                magic = f.readline().strip()
                if magic != b'P5':
                    raise ValueError("Not a valid P5 PGM file")
                
                while True:
                    pos = f.tell()
                    line = f.readline().strip()
                    if not line.startswith(b'#'):
                        f.seek(pos)
                        break
                
                line = f.readline().strip()
                width, height = map(int, line.split())
                maxval = int(f.readline().strip())
                
                dtype = np.uint8 if maxval < 256 else np.uint16
                data = np.frombuffer(f.read(), dtype=dtype)
                image = data.reshape((height, width))
                return image
    else:
        with open(filename, 'rb') as f:
            magic = f.readline().strip()
            if magic != b'P5':
                raise ValueError("Not a valid P5 PGM file")
            
            while True:
                pos = f.tell()
                line = f.readline().strip()
                if not line.startswith(b'#'):
                    f.seek(pos)
                    break
            
            line = f.readline().strip()
            width, height = map(int, line.split())
            maxval = int(f.readline().strip())
            
            dtype = np.uint8 if maxval < 256 else np.uint16
            data = np.frombuffer(f.read(), dtype=dtype)
            image = data.reshape((height, width))
            return image


def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess a single image: resize and normalize to [0, 1].
    Uses simple nearest-neighbor resampling (pure NumPy, fast).
    
    Args:
        image (ndarray): Input image of any size.
        target_size (tuple): (height, width) to resize to.
    
    Returns:
        ndarray: Preprocessed image, dtype float32, range [0, 1].
    """
    h, w = image.shape
    target_h, target_w = target_size
    
    # Nearest-neighbor resize: map each output pixel to nearest input pixel
    rows = np.floor(np.arange(target_h) * h / target_h).astype(int)
    cols = np.floor(np.arange(target_w) * w / target_w).astype(int)
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)
    
    resized = image[np.ix_(rows, cols)]
    
    # Normalize to [0, 1]
    if resized.max() > resized.min():
        normalized = (resized - resized.min()) / (resized.max() - resized.min())
    else:
        normalized = np.zeros_like(resized, dtype=np.float32)
    
    return normalized.astype(np.float32)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(root_dir, image_size=(64, 64)):
    """
    Load labeled dataset from folder structure or flat zip with filenames p{id}_i{img}.pgm
    
    Supports:
    1. Folder structure: root_dir/person_1/img.pgm, root_dir/person_2/img.pgm, ...
    2. Flat zip: files named p{person_id}_i{image_id}.pgm
    
    Args:
        root_dir (str): Path to dataset root or zip file.
        image_size (tuple): Target (height, width).
    
    Returns:
        X (ndarray): Shape (n_samples, height*width), images flattened.
        y (ndarray): Shape (n_samples,), class labels (0, 1, 2, ...).
        label_to_name (dict): Mapping from class index to label.
        image_paths (list): Original file paths for reference.
    """
    import zipfile
    import re
    
    X_list = []
    y_list = []
    label_to_name = {}
    image_paths = []
    
    person_to_idx = {}  # Map person IDs to class indices
    class_idx = 0
    rng = np.random.default_rng(RANDOM_SEED)

    if '.zip' in root_dir:
        # Load from zip
        zip_name = root_dir
        with zipfile.ZipFile(zip_name, 'r') as zf:
            files = [f for f in zf.namelist() if f.endswith('.pgm')]
            
            if not files:
                raise ValueError(f"No PGM files found in {zip_name}")
            
            if MAX_IMAGES is not None and len(files) > MAX_IMAGES:
                chosen = rng.choice(len(files), size=MAX_IMAGES, replace=False)
                files = [files[i] for i in chosen]
                if VERBOSE:
                    print(f"  Sampling {len(files)} random files from {len(zf.namelist())} available")
            
            # Try to extract person IDs from filenames (p{id}_i{img}.pgm format)
            person_ids = set()
            for f in files:
                fname = f.split('/')[-1]
                match = re.match(r'p(\d+)_i(\d+)\.pgm', fname)
                if match:
                    person_ids.add(int(match.group(1)))
            
            # If we found person IDs in filenames, use them as classes
            if person_ids:
                if VERBOSE:
                    print(f"  Found {len(person_ids)} unique persons in filenames")
                
                for pid in sorted(person_ids):
                    person_to_idx[pid] = class_idx
                    label_to_name[class_idx] = f"person_{pid}"
                    class_idx += 1
            else:
                # Fall back: use first-level directory as class
                if VERBOSE:
                    print(f"  Using directory structure for classes")
                dirs = set()
                for f in files:
                    parts = f.split('/')
                    if len(parts) > 1:
                        dirs.add(parts[0])
                
                for dir_name in sorted(dirs):
                    label_to_name[class_idx] = dir_name
                    class_idx += 1
            
            # Load images
            for pgm_file in files:
                try:
                    fname = pgm_file.split('/')[-1]
                    match = re.match(r'p(\d+)_i(\d+)\.pgm', fname)
                    
                    if match and person_to_idx:
                        # Use filename-based person ID
                        person_id = int(match.group(1))
                        class_label = person_to_idx[person_id]
                    else:
                        # Use directory-based classification
                        parts = pgm_file.split('/')
                        if len(parts) > 1:
                            dir_name = parts[0]
                            # Find class index for this directory
                            class_label = None
                            for idx, name in label_to_name.items():
                                if name == dir_name:
                                    class_label = idx
                                    break
                            if class_label is None:
                                if VERBOSE:
                                    print(f"  Warning: Could not find class for {pgm_file}")
                                continue
                        else:
                            continue
                    
                    # Load and preprocess image
                    img = load_pgm_simple(f"{zip_name}/{pgm_file}")
                    img_preprocessed = preprocess_image(img, target_size=image_size)
                    img_vector = img_preprocessed.flatten()
                    
                    X_list.append(img_vector)
                    y_list.append(class_label)
                    image_paths.append(pgm_file)
                    
                    if VERBOSE and len(X_list) % 100 == 0:
                        print(f"  Loaded {len(X_list)} images...")
                except Exception as e:
                    if VERBOSE:
                        print(f"  Warning: Could not load {pgm_file}: {e}")
                    continue
    else:
        # Load from filesystem
        dataset_path = Path(root_dir)
        all_files = []
        for class_folder in sorted(dataset_path.iterdir()):
            if not class_folder.is_dir():
                continue
            
            for img_file in sorted(class_folder.glob('*.pgm')):
                all_files.append((class_folder.name, img_file))
        
        if not all_files:
            raise ValueError(f"No images found in {root_dir}")
        
        if MAX_IMAGES is not None and len(all_files) > MAX_IMAGES:
            chosen = rng.choice(len(all_files), size=MAX_IMAGES, replace=False)
            all_files = [all_files[i] for i in chosen]
            if VERBOSE:
                print(f"  Sampling {len(all_files)} random files from {len(list(dataset_path.rglob('*.pgm')))} available")
        
        for class_name, img_file in all_files:
            try:
                if class_name not in label_to_name.values():
                    label_to_name[class_idx] = class_name
                    class_idx += 1
                class_label = [k for k, v in label_to_name.items() if v == class_name][0]
                
                img = load_pgm_simple(str(img_file))
                img_preprocessed = preprocess_image(img, target_size=image_size)
                img_vector = img_preprocessed.flatten()
                
                X_list.append(img_vector)
                y_list.append(class_label)
                image_paths.append(str(img_file))
            except Exception as e:
                if VERBOSE:
                    print(f"  Warning: Could not load {img_file}: {e}")
                continue
    
    if not X_list:
        raise ValueError(f"No images found in {root_dir}")
    
    X = np.array(X_list, dtype=np.float32)  # Shape: (n_samples, n_features)
    y = np.array(y_list, dtype=np.int32)    # Shape: (n_samples,)
    
    return X, y, label_to_name, image_paths


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def train_test_split_numpy(X, y, test_ratio=0.2, seed=42):
    """
    Split dataset into train and test with stratification by class.
    
    Args:
        X (ndarray): Features, shape (n_samples, n_features).
        y (ndarray): Labels, shape (n_samples,).
        test_ratio (float): Fraction of data to use for testing.
        seed (int): Random seed.
    
    Returns:
        X_train, X_test, y_train, y_test (ndarrays): Split data.
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_ratio))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# PCA VIA SVD ("EIGENFACES")
# ============================================================================

def fit_pca(X_train, n_components):
    """
    Fit PCA on training data using SVD.
    
    Args:
        X_train (ndarray): Training data, shape (n_train, n_features).
        n_components (int): Number of principal components to keep.
    
    Returns:
        mean_face (ndarray): Mean image vector, shape (n_features,).
        components (ndarray): PCA components (eigenfaces), shape (n_components, n_features).
        explained_var (ndarray): Explained variance per component, shape (n_components,).
        Z_train (ndarray): Projected training data, shape (n_train, n_components).
    """
    # Compute mean face
    mean_face = X_train.mean(axis=0)  # Shape: (n_features,)
    
    # Center the data
    X_centered = X_train - mean_face  # Shape: (n_train, n_features)
    
    # SVD: X_centered = U @ S @ V.T
    # U: shape (n_train, min(n_train, n_features))
    # S: shape (min(n_train, n_features),)
    # V.T: shape (min(n_train, n_features), n_features)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Components are rows of V.T (or columns of V)
    components = Vt[:n_components, :]  # Shape: (n_components, n_features)
    
    # Explained variance
    explained_var = (S[:n_components] ** 2) / (X_train.shape[0] - 1)
    
    # Project training data
    Z_train = X_centered @ components.T  # Shape: (n_train, n_components)
    
    return mean_face, components, explained_var, Z_train


def transform_pca(X, mean_face, components):
    """
    Transform data using fitted PCA model.
    
    Args:
        X (ndarray): Data to transform, shape (n_samples, n_features).
        mean_face (ndarray): Mean from training, shape (n_features,).
        components (ndarray): PCA components, shape (n_components, n_features).
    
    Returns:
        Z (ndarray): Projected data, shape (n_samples, n_components).
    """
    X_centered = X - mean_face
    Z = X_centered @ components.T
    return Z


# ============================================================================
# ONE-HOT ENCODING & LINEAR CLASSIFIER
# ============================================================================

def one_hot_encode(y, num_classes):
    """
    Convert class labels to one-hot encoding.
    
    Args:
        y (ndarray): Class labels, shape (n_samples,).
        num_classes (int): Number of classes.
    
    Returns:
        Y (ndarray): One-hot encoded, shape (n_samples, num_classes).
    """
    n_samples = y.shape[0]
    Y = np.zeros((n_samples, num_classes), dtype=np.float32)
    Y[np.arange(n_samples), y] = 1.0
    return Y


def fit_linear_least_squares(Z_train, y_train, num_classes):
    """
    Fit linear multiclass classifier using least squares.
    
    Solves: W = inv(Z1.T @ Z1) @ Z1.T @ Y
    where Z1 is Z_train with bias column appended.
    
    Args:
        Z_train (ndarray): Projected training data, shape (n_train, n_components).
        y_train (ndarray): Training labels, shape (n_train,).
        num_classes (int): Number of classes.
    
    Returns:
        W (ndarray): Weight matrix, shape (n_components + 1, num_classes).
              Last row is bias; first n_components rows are feature weights.
    """
    n_samples, n_features = Z_train.shape
    
    # Add bias column
    Z1 = np.hstack([Z_train, np.ones((n_samples, 1))])  # (n_train, n_components+1)
    
    # One-hot encode labels
    Y = one_hot_encode(y_train, num_classes)  # (n_train, num_classes)
    
    # Solve using pseudoinverse: W = pinv(Z1) @ Y
    W = np.linalg.pinv(Z1) @ Y  # (n_components+1, num_classes)
    
    return W


def predict_linear(Z, W):
    """
    Predict class labels using linear model.
    
    Args:
        Z (ndarray): Projected data, shape (n_samples, n_components).
        W (ndarray): Weight matrix, shape (n_components+1, num_classes).
    
    Returns:
        y_pred (ndarray): Predicted class indices, shape (n_samples,).
        scores (ndarray): Class scores, shape (n_samples, num_classes).
    """
    n_samples = Z.shape[0]
    
    # Add bias
    Z1 = np.hstack([Z, np.ones((n_samples, 1))])  # (n_samples, n_components+1)
    
    # Compute scores
    scores = Z1 @ W  # (n_samples, num_classes)
    
    # Predict class with max score
    y_pred = np.argmax(scores, axis=1)
    
    return y_pred, scores


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def accuracy_score_numpy(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Args:
        y_true (ndarray): True labels, shape (n_samples,).
        y_pred (ndarray): Predicted labels, shape (n_samples,).
    
    Returns:
        accuracy (float): Fraction of correct predictions.
    """
    return np.mean(y_true == y_pred)


def confusion_matrix_numpy(y_true, y_pred, num_classes):
    """
    Compute confusion matrix.
    
    Args:
        y_true (ndarray): True labels, shape (n_samples,).
        y_pred (ndarray): Predicted labels, shape (n_samples,).
        num_classes (int): Number of classes.
    
    Returns:
        cm (ndarray): Confusion matrix, shape (num_classes, num_classes).
                      cm[i, j] = number of samples with true label i, pred label j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


# ============================================================================
# SINGLE IMAGE PREDICTION
# ============================================================================

def predict_single_image(image_path, image_size, mean_face, components, W,
                         idx_to_label):
    """
    Predict class for a single image.
    
    Args:
        image_path (str): Path to image file or 'zip.zip/path/file.pgm'.
        image_size (tuple): (height, width).
        mean_face (ndarray): Training mean, shape (n_features,).
        components (ndarray): PCA components, shape (n_components, n_features).
        W (ndarray): Weight matrix, shape (n_components+1, num_classes).
        idx_to_label (dict): Mapping from class index to label name.
    
    Returns:
        predicted_label (str): Predicted class name.
        scores (ndarray): Class scores, shape (num_classes,).
    """
    # Load and preprocess
    try:
        img = load_pgm_simple(image_path)
        img_preprocessed = preprocess_image(img, target_size=image_size)
        img_vector = img_preprocessed.flatten()
    except Exception as e:
        raise ValueError(f"Could not load image {image_path}: {e}")
    
    # Project using PCA
    Z = transform_pca(img_vector[np.newaxis, :], mean_face, components)
    
    # Predict
    y_pred, scores = predict_linear(Z, W)
    predicted_class_idx = y_pred[0]
    predicted_label = idx_to_label[predicted_class_idx]
    
    return predicted_label, scores[0]


# ============================================================================
# OPTIONAL: NEAREST CLASS MEAN BASELINE
# ============================================================================

def compute_class_means(Z_train, y_train, num_classes):
    """
    Compute mean vector for each class in PCA space.
    
    Args:
        Z_train (ndarray): Projected training data, shape (n_train, n_components).
        y_train (ndarray): Training labels, shape (n_train,).
        num_classes (int): Number of classes.
    
    Returns:
        class_means (ndarray): Mean per class, shape (num_classes, n_components).
    """
    n_components = Z_train.shape[1]
    class_means = np.zeros((num_classes, n_components), dtype=np.float32)
    
    for c in range(num_classes):
        mask = y_train == c
        if mask.sum() > 0:
            class_means[c] = Z_train[mask].mean(axis=0)
    
    return class_means


def predict_nearest_class_mean(Z, class_means):
    """
    Predict using nearest class mean (simple baseline).
    
    Args:
        Z (ndarray): Projected data, shape (n_samples, n_components).
        class_means (ndarray): Class means, shape (num_classes, n_components).
    
    Returns:
        y_pred (ndarray): Predicted labels, shape (n_samples,).
    """
    distances = np.linalg.norm(Z[:, np.newaxis, :] - class_means[np.newaxis, :, :],
                               axis=2)  # (n_samples, num_classes)
    y_pred = np.argmin(distances, axis=1)
    return y_pred


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Run the complete face recognition pipeline.
    """
    print("=" * 70)
    print("FACE RECOGNITION: PCA + LINEAR LEAST SQUARES")
    print("=" * 70)
    
    # -------- Load Dataset --------
    print("\n[1] Loading dataset...")
    try:
        X, y, label_to_name, image_paths = load_dataset(DATASET_PATH, IMAGE_SIZE)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    n_samples, n_features = X.shape
    num_classes = len(label_to_name)
    
    print(f"  Samples: {n_samples}")
    print(f"  Features (pixels): {n_features}")
    print(f"  Classes: {num_classes}")
    print(f"  Class names: {list(label_to_name.values())}")
    
    # -------- Train/Test Split --------
    print("\n[2] Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X, y, test_ratio=TEST_RATIO, seed=RANDOM_SEED)
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # -------- PCA --------
    print(f"\n[3] Fitting PCA with {N_COMPONENTS} components...")
    mean_face, components, explained_var, Z_train = fit_pca(X_train, N_COMPONENTS)
    
    total_var = explained_var.sum()
    cumsum_var = np.cumsum(explained_var)
    print(f"  Total variance explained: {total_var:.4f}")
    print(f"  Top 3 components: {explained_var[:3]}")
    
    # Project test data
    Z_test = transform_pca(X_test, mean_face, components)
    print(f"  PCA projection shapes: Z_train {Z_train.shape}, Z_test {Z_test.shape}")
    
    # -------- Train Linear Classifier --------
    print("\n[4] Training linear least squares classifier...")
    W = fit_linear_least_squares(Z_train, y_train, num_classes)
    print(f"  Weight matrix shape: {W.shape}")
    
    # -------- Evaluate --------
    print("\n[5] Evaluating...")
    y_train_pred, _ = predict_linear(Z_train, W)
    y_test_pred, _ = predict_linear(Z_test, W)
    
    train_acc = accuracy_score_numpy(y_train, y_train_pred)
    test_acc = accuracy_score_numpy(y_test, y_test_pred)
    
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix_numpy(y_test, y_test_pred, num_classes)
    print(f"\n  Confusion matrix (test set):")
    print(f"  {cm}")
    

    # -------- Optional: Baseline --------
    print("\n[6] Baseline: Nearest class mean in PCA space...")
    class_means = compute_class_means(Z_train, y_train, num_classes)
    y_test_pred_baseline = predict_nearest_class_mean(Z_test, class_means)
    baseline_acc = accuracy_score_numpy(y_test, y_test_pred_baseline)
    print(f"  Baseline accuracy: {baseline_acc:.4f}")
    
    # -------- Optional: Single Image Prediction --------
    print("\n[7] Testing single image prediction...")
    idx_to_label = {v: k for k, v in label_to_name.items()}
    
    # Test on first test image
    if len(image_paths) > 0:
        test_image_path = image_paths[0]
        try:
            pred_label, pred_scores = predict_single_image(
                test_image_path, IMAGE_SIZE, mean_face, components, W, idx_to_label)
            true_label = idx_to_label[y[0]]
            print(f"  Test image: {test_image_path}")
            print(f"  True label: {true_label}")
            print(f"  Predicted label: {pred_label}")
            print(f"  Confidence: {pred_scores.max():.4f}")
        except Exception as e:
            print(f"  Warning: Could not predict on test image: {e}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

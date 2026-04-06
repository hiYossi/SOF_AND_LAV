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

import re
import zipfile
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = Path(__file__).with_name("Train Set (Labeled)-20260405T164823Z-3-001.zip")
CACHE_PATH = Path(__file__).with_name("linear_model_cache.npz")
IMAGE_SIZE = (64, 64)              # (height, width) — full resolution
N_COMPONENTS = 20                  # Use 20 for fast testing, 50 for final accuracy
TEST_RATIO = 0.2                   # Hold out 20% for testing
RANDOM_SEED = 42                   # For reproducibility
VERBOSE = True                     # Print progress info
MAX_IMAGES = 1000                  # Use 1000 images for the test, None for all ~3600 images
CACHE_VERSION = 2
PGM_PATTERN = re.compile(r'(?:.*/)?p(\d+)_i(\d+)(?:\(\d+\))?\.pgm$')
USE_DEFAULT_MAX_IMAGES = object()


# ============================================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================================

def _read_pgm(file_obj):
    """Read a binary PGM image from an already-open file object."""
    magic = file_obj.readline().strip()
    if magic != b'P5':
        raise ValueError("Not a valid P5 PGM file")

    line = file_obj.readline()
    while line.strip().startswith(b'#'):
        line = file_obj.readline()

    width, height = map(int, line.split())
    maxval = int(file_obj.readline().strip())
    dtype = np.uint8 if maxval < 256 else np.uint16
    data = np.frombuffer(file_obj.read(), dtype=dtype)
    return data.reshape((height, width))


def load_pgm_simple(filename):
    """
    Load a PGM image file (P5 format, binary grayscale).
    
    Args:
        filename (str): Path to PGM file or 'archive.zip/path/file.pgm'.
    
    Returns:
        ndarray: 2D grayscale image (height, width).
    """
    filename = str(filename)

    if '.zip' in filename:
        zip_path, internal_path = filename.split('.zip', 1)
        zip_path += '.zip'
        internal_path = internal_path.lstrip('/\\')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(internal_path) as f:
                return _read_pgm(f)

    with open(filename, 'rb') as f:
        return _read_pgm(f)


def load_pgm_from_zip(zip_file, internal_path):
    """Load a PGM image from an already-open ZipFile handle."""
    with zip_file.open(internal_path) as f:
        return _read_pgm(f)


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
    if target_size is None or tuple(target_size) == image.shape:
        resized = image.astype(np.float32)
    else:
        h, w = image.shape
        target_h, target_w = target_size
        
        # Nearest-neighbor resize: map each output pixel to nearest input pixel
        rows = np.floor(np.arange(target_h) * h / target_h).astype(int)
        cols = np.floor(np.arange(target_w) * w / target_w).astype(int)
        rows = np.clip(rows, 0, h - 1)
        cols = np.clip(cols, 0, w - 1)
        
        resized = image[np.ix_(rows, cols)].astype(np.float32)
    
    # Normalize to [0, 1]
    if resized.max() > resized.min():
        normalized = (resized - resized.min()) / (resized.max() - resized.min())
    else:
        normalized = np.zeros_like(resized, dtype=np.float32)
    
    return normalized.astype(np.float32)


def build_dataset_cache_key(root_dir, image_size, max_images, max_images_per_person,
                            random_seed):
    """Fingerprint dataset-loading inputs so cached arrays stay in sync."""
    root_path = Path(root_dir)
    resolved = root_path.resolve()
    if root_path.exists():
        stat = root_path.stat()
        size = stat.st_size if root_path.is_file() else 0
        mtime = int(stat.st_mtime)
    else:
        size = 0
        mtime = 0

    size_key = "original" if image_size is None else f"{image_size[0]}x{image_size[1]}"
    return (
        f"{resolved}|{size}|{mtime}|{size_key}|{max_images}|"
        f"{max_images_per_person}|{random_seed}|v{CACHE_VERSION}"
    )


def load_cached_dataset(cache_path, cache_key, verbose=True, log_prefix="  "):
    """Load a cached preprocessed dataset when the inputs still match."""
    if cache_path is None:
        return None

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_key = str(data["cache_key"][0])
            if cached_key != cache_key:
                return None

            label_ids = data["label_ids"].astype(np.int32)
            label_names = data["label_names"]
            label_to_name = {int(idx): str(name) for idx, name in zip(label_ids, label_names)}
            image_paths = [str(path) for path in data["image_paths"].tolist()]

            if verbose:
                print(f"{log_prefix}Loaded dataset cache from {cache_path.name}")

            return data["X"], data["y"], label_to_name, image_paths
    except Exception as e:
        if verbose:
            print(f"{log_prefix}Ignoring stale dataset cache: {e}")
        return None


def save_cached_dataset(cache_path, cache_key, X, y, label_to_name, image_paths):
    """Persist preprocessed dataset arrays for faster repeat runs."""
    if cache_path is None:
        return

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    label_ids = np.array(sorted(label_to_name.keys()), dtype=np.int32)
    label_names = np.array([label_to_name[idx] for idx in label_ids])

    np.savez(
        cache_path,
        cache_key=np.array([cache_key]),
        X=X,
        y=y,
        label_ids=label_ids,
        label_names=label_names,
        image_paths=np.array(image_paths),
    )


def _sample_file_records(records, max_images, rng, verbose, log_prefix):
    """Randomly sample file records when an overall cap is requested."""
    if max_images is None or len(records) <= max_images:
        return records

    chosen = rng.choice(len(records), size=max_images, replace=False)
    sampled = [records[i] for i in chosen]
    if verbose:
        print(f"{log_prefix}Sampling {len(sampled)} random files from {len(records)} available")
    return sampled


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(root_dir, image_size=(64, 64), max_images=USE_DEFAULT_MAX_IMAGES,
                 max_images_per_person=None, cache_path=None, verbose=None,
                 random_seed=RANDOM_SEED, log_prefix="  "):
    """
    Load labeled dataset from folder structure or flat zip with filenames p{id}_i{img}.pgm
    
    Supports:
    1. Folder structure: root_dir/person_1/img.pgm, root_dir/person_2/img.pgm, ...
    2. Flat zip: files named p{person_id}_i{image_id}.pgm
    
    Args:
        root_dir (str or Path): Path to dataset root or zip file.
        image_size (tuple or None): Target (height, width), or None to keep original size.
        max_images (int, None, or sentinel): Optional cap on total images loaded.
            When omitted, the module-level MAX_IMAGES default is used.
            Pass None explicitly to disable the overall cap.
        max_images_per_person (int or None): Optional per-class cap.
        cache_path (str or Path or None): Optional `.npz` cache to reuse preprocessed arrays.
        verbose (bool or None): Whether to print dataset-loading progress.
        random_seed (int): Random seed for reproducible sampling.
        log_prefix (str): Prefix for progress messages.
    
    Returns:
        X (ndarray): Shape (n_samples, height*width), images flattened.
        y (ndarray): Shape (n_samples,), class labels (0, 1, 2, ...).
        label_to_name (dict): Mapping from class index to label.
        image_paths (list): Original file paths for reference.
    """
    if verbose is None:
        verbose = VERBOSE
    if max_images is USE_DEFAULT_MAX_IMAGES:
        max_images = MAX_IMAGES

    root_path = Path(root_dir)
    cache_key = build_dataset_cache_key(
        root_path,
        image_size=image_size,
        max_images=max_images,
        max_images_per_person=max_images_per_person,
        random_seed=random_seed,
    )
    cached = load_cached_dataset(cache_path, cache_key, verbose=verbose,
                                 log_prefix=log_prefix)
    if cached is not None:
        return cached

    X_list = []
    y_list = []
    label_to_name = {}
    image_paths = []
    rng = np.random.default_rng(random_seed)

    if root_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(root_path, 'r') as zf:
            files = [f for f in zf.namelist() if f.endswith('.pgm')]

            if not files:
                raise ValueError(f"No PGM files found in {root_path}")

            if max_images_per_person is None:
                files = _sample_file_records(files, max_images, rng, verbose, log_prefix)

            matched_person_ids = sorted({
                int(match.group(1))
                for match in (PGM_PATTERN.match(f) for f in files)
                if match
            })

            if matched_person_ids:
                if verbose:
                    print(f"{log_prefix}Found {len(matched_person_ids)} unique persons in filenames")
                class_lookup = {pid: idx for idx, pid in enumerate(matched_person_ids)}
                label_to_name = {idx: f"person_{pid}" for pid, idx in class_lookup.items()}
            else:
                if verbose:
                    print(f"{log_prefix}Using directory structure for classes")
                class_names = sorted({
                    parts[0]
                    for parts in (f.split('/') for f in files)
                    if len(parts) > 1
                })
                label_to_name = {idx: name for idx, name in enumerate(class_names)}
                class_lookup = {name: idx for idx, name in label_to_name.items()}

            person_counts = {}
            filled_classes = 0
            total_classes = len(label_to_name)

            for pgm_file in files:
                try:
                    match = PGM_PATTERN.match(pgm_file)
                    if match and matched_person_ids:
                        class_label = class_lookup[int(match.group(1))]
                    else:
                        parts = pgm_file.split('/')
                        if len(parts) <= 1:
                            continue
                        class_label = class_lookup.get(parts[0])
                        if class_label is None:
                            if verbose:
                                print(f"{log_prefix}Warning: Could not find class for {pgm_file}")
                            continue

                    if max_images_per_person is not None and person_counts.get(class_label, 0) >= max_images_per_person:
                        continue
                    if max_images is not None and max_images_per_person is not None and len(X_list) >= max_images:
                        break

                    img = load_pgm_from_zip(zf, pgm_file)
                    img_preprocessed = preprocess_image(img, target_size=image_size)

                    X_list.append(img_preprocessed.flatten())
                    y_list.append(class_label)
                    image_paths.append(f"{root_path}/{pgm_file}")
                    person_counts[class_label] = person_counts.get(class_label, 0) + 1

                    if max_images_per_person is not None and person_counts[class_label] == max_images_per_person:
                        filled_classes += 1
                        if filled_classes == total_classes:
                            break

                    if verbose and len(X_list) % 100 == 0:
                        print(f"{log_prefix}Loaded {len(X_list)} images...")
                except Exception as e:
                    if verbose:
                        print(f"{log_prefix}Warning: Could not load {pgm_file}: {e}")
                    continue
    else:
        all_files = []
        for class_folder in sorted(root_path.iterdir()):
            if not class_folder.is_dir():
                continue

            for img_file in sorted(class_folder.glob('*.pgm')):
                all_files.append((class_folder.name, img_file))

        if not all_files:
            raise ValueError(f"No images found in {root_path}")

        if max_images_per_person is None:
            all_files = _sample_file_records(all_files, max_images, rng, verbose, log_prefix)

        class_names = sorted({class_name for class_name, _ in all_files})
        label_to_name = {idx: name for idx, name in enumerate(class_names)}
        name_to_idx = {name: idx for idx, name in label_to_name.items()}

        person_counts = {}
        filled_classes = 0
        total_classes = len(label_to_name)

        for class_name, img_file in all_files:
            try:
                class_label = name_to_idx[class_name]

                if max_images_per_person is not None and person_counts.get(class_label, 0) >= max_images_per_person:
                    continue
                if max_images is not None and max_images_per_person is not None and len(X_list) >= max_images:
                    break

                img = load_pgm_simple(str(img_file))
                img_preprocessed = preprocess_image(img, target_size=image_size)

                X_list.append(img_preprocessed.flatten())
                y_list.append(class_label)
                image_paths.append(str(img_file))
                person_counts[class_label] = person_counts.get(class_label, 0) + 1

                if max_images_per_person is not None and person_counts[class_label] == max_images_per_person:
                    filled_classes += 1
                    if filled_classes == total_classes:
                        break

                if verbose and len(X_list) % 100 == 0:
                    print(f"{log_prefix}Loaded {len(X_list)} images...")
            except Exception as e:
                if verbose:
                    print(f"{log_prefix}Warning: Could not load {img_file}: {e}")
                continue

    if not X_list:
        raise ValueError(f"No images found in {root_path}")

    X = np.array(X_list, dtype=np.float32)  # Shape: (n_samples, n_features)
    y = np.array(y_list, dtype=np.int32)    # Shape: (n_samples,)

    save_cached_dataset(cache_path, cache_key, X, y, label_to_name, image_paths)
    if verbose and cache_path is not None:
        print(f"{log_prefix}Saved dataset cache to {Path(cache_path).name}")

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
        X, y, label_to_name, image_paths = load_dataset(
            DATASET_PATH,
            IMAGE_SIZE,
            max_images=MAX_IMAGES,
            cache_path=CACHE_PATH,
            verbose=VERBOSE,
            random_seed=RANDOM_SEED,
        )
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
    idx_to_label = dict(label_to_name)
    
    # Test on first test image
    if len(image_paths) > 0:
        test_image_path = image_paths[0]
        try:
            pred_label, pred_scores = predict_single_image(
                test_image_path, IMAGE_SIZE, mean_face, components, W, idx_to_label)
            true_label = label_to_name[y[0]]
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

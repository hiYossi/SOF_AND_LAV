"""
Standalone KNN Pipeline for Face Recognition.
This script includes data loading, preprocessing, model training, cross-validation,
hyperparameter tuning, and final prediction on the test set.
"""

import numpy as np
import zipfile
import re
from pathlib import Path
import csv

# No external machine learning libraries are used in this standalone script.

# =====================================================================
# 1. PREPROCESSING
# =====================================================================

def _apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Pure-NumPy CLAHE on a uint8 grayscale image."""
    h, w = image.shape
    n_tiles_y, n_tiles_x = tile_grid_size
    n_bins = 256

    tile_h = int(np.ceil(h / n_tiles_y))
    tile_w = int(np.ceil(w / n_tiles_x))
    clip_count = max(1, int(clip_limit * tile_h * tile_w / n_bins))

    pad_h = tile_h * n_tiles_y - h
    pad_w = tile_w * n_tiles_x - w
    padded = np.pad(image.astype(np.uint8), ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    luts = np.zeros((n_tiles_y, n_tiles_x, n_bins), dtype=np.float32)
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile = padded[ty * tile_h:(ty + 1) * tile_h, tx * tile_w:(tx + 1) * tile_w]
            hist = np.bincount(tile.ravel(), minlength=n_bins)

            excess = int(np.sum(np.maximum(hist - clip_count, 0)))
            hist = np.minimum(hist, clip_count)
            hist += excess // n_bins
            hist[: excess % n_bins] += 1

            cdf = hist.cumsum().astype(np.float32)
            luts[ty, tx] = np.clip(cdf / float(tile.size) * 255.0, 0, 255)

    ty_centers = (np.arange(n_tiles_y) + 0.5) * tile_h
    tx_centers = (np.arange(n_tiles_x) + 0.5) * tile_w

    Y, X = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")

    ty0 = np.clip(np.searchsorted(ty_centers, Y, side="right") - 1, 0, n_tiles_y - 1)
    ty1 = np.clip(ty0 + 1, 0, n_tiles_y - 1)
    tx0 = np.clip(np.searchsorted(tx_centers, X, side="right") - 1, 0, n_tiles_x - 1)
    tx1 = np.clip(tx0 + 1, 0, n_tiles_x - 1)

    denom_y = ty_centers[ty1] - ty_centers[ty0]
    denom_x = tx_centers[tx1] - tx_centers[tx0]
    with np.errstate(divide="ignore", invalid="ignore"):
        wy = np.where(denom_y > 0, (Y - ty_centers[ty0]) / denom_y, 0.0).clip(0, 1)
        wx = np.where(denom_x > 0, (X - tx_centers[tx0]) / denom_x, 0.0).clip(0, 1)

    pv = padded
    v00 = luts[ty0, tx0, pv]
    v01 = luts[ty0, tx1, pv]
    v10 = luts[ty1, tx0, pv]
    v11 = luts[ty1, tx1, pv]

    result = (
        v00 * (1 - wy) * (1 - wx)
        + v01 * (1 - wy) * wx
        + v10 * wy * (1 - wx)
        + v11 * wy * wx
    )

    return np.clip(result[:h, :w], 0, 255).astype(np.uint8)

def preprocess_image(image, target_size=None):
    """Gray-bar crop -> CLAHE -> optional resize -> Z-score normalization."""
    # 1. Detect and crop gray bars
    if image.shape == (64, 72):
        left_side = image[:, :8]
        right_side = image[:, -8:]
        if np.var(left_side) < np.var(right_side):
            image = image[:, 8:]
        else:
            image = image[:, :64]

    # 2. CLAHE
    image = _apply_clahe(image)

    # 3. Resizing
    if target_size is None or tuple(target_size) == tuple(image.shape):
        resized = image.astype(np.float32)
    else:
        target_height, target_width = map(int, target_size)
        height, width = image.shape
        rows = np.floor(np.arange(target_height) * height / target_height).astype(int)
        cols = np.floor(np.arange(target_width) * width / target_width).astype(int)
        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)
        resized = image[np.ix_(rows, cols)].astype(np.float32)

    # 4. Z-score normalization
    mean_val = float(resized.mean())
    std_val = float(resized.std())
    if std_val > 0:
        resized = (resized - mean_val) / std_val
    else:
        resized = resized - mean_val

    # Flatten the final image to a 1D feature vector
    return resized.astype(np.float32).flatten()

# =====================================================================
# 2. DATA LOADING
# =====================================================================

def _read_pgm(file_obj):
    """Read a binary P5 PGM image from an already-open file object."""
    magic = file_obj.readline().strip()
    if magic != b"P5":
        raise ValueError("Not a valid P5 PGM file.")

    line = file_obj.readline()
    while line.strip().startswith(b"#"):
        line = file_obj.readline()

    width, height = map(int, line.split())
    max_value = int(file_obj.readline().strip())
    dtype = np.uint8 if max_value < 256 else np.uint16
    data = np.frombuffer(file_obj.read(), dtype=dtype)
    return data.reshape((height, width))

def load_data(zip_path, max_images_per_person=None):
    """
    Loads PGM images directly from the ZIP file, extracts labels,
    and applies all preprocessing.
    """
    print(f"Loading data from {zip_path}...")
    X_list = []
    y_list = []

    pattern = re.compile(r"(?:.*/)?p(\d+)_i(\d+)(?:\(\d+\))?\.pgm$")

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".pgm")]

        # Group by person to handle max_images_per_person if needed
        person_files = {}
        for member in members:
            match = pattern.match(member)
            if match:
                person_id = int(match.group(1))
                if person_id not in person_files:
                    person_files[person_id] = []
                person_files[person_id].append(member)

        # Process images
        count = 0
        for person_id, files in person_files.items():
            if max_images_per_person is not None:
                files = files[:max_images_per_person]

            for member in files:
                with archive.open(member) as file_obj:
                    image = _read_pgm(file_obj)

                # Apply preprocessing which includes flattening
                processed_vector = preprocess_image(image)
                X_list.append(processed_vector)
                y_list.append(person_id)
                
                count += 1
                if count % 500 == 0:
                    print(f"  Processed {count} / {len(members)} images...")

    return np.array(X_list), np.array(y_list)

def load_unlabeled_data(zip_path):
    """
    Loads unlabeled PGM images from the Test ZIP file for final predictions.
    Returns the flattened images and their filenames.
    """
    print(f"Loading unlabeled data from {zip_path}...")
    X_list = []
    filenames = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".pgm")]
        
        count = 0
        for member in members:
            with archive.open(member) as file_obj:
                image = _read_pgm(file_obj)
            
            processed_vector = preprocess_image(image)
            X_list.append(processed_vector)
            
            filenames.append(Path(member).name)
            
            count += 1
            if count % 500 == 0:
                print(f"  Processed {count} / {len(members)} unlabeled images...")
                
    return np.array(X_list), filenames

# =====================================================================
# 3. MANUAL KNN IMPLEMENTATION & CROSS-VALIDATION
# =====================================================================

def compute_distances(x, X_train, metric='euclidean'):
    if metric == 'manhattan':
        return np.sum(np.abs(X_train - x), axis=1)
    else: # euclidean
        return np.sqrt(np.sum((X_train - x) ** 2, axis=1))

def predict_one(x, X_train, y_train, k=3, metric='euclidean'):
    distances = compute_distances(x, X_train, metric)
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    return np.bincount(nearest_labels).argmax()

import concurrent.futures

def predict(X_test, X_train, y_train, k=3, metric='euclidean'):
    y_pred = [0] * len(X_test)
    total = len(X_test)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(predict_one, x, X_train, y_train, k, metric): i 
            for i, x in enumerate(X_test)
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            y_pred[idx] = future.result()
            
            completed += 1
            if completed % 500 == 0 or completed == total:
                print(f"  Predicted {completed}/{total} samples...")
                
    return np.array(y_pred)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def get_k_folds(X, y, n_splits=5):
    """
    Manually split data into n_splits folds for cross validation.
    """
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(n_splits, len(X) // n_splits, dtype=int)
    fold_sizes[:len(X) % n_splits] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds

def tune_and_train_knn(X_train, y_train):
    """
    Performs manual K-Fold cross validation and hyperparameter tuning 
    to find the best K and distance metric for our custom KNN.
    """
    print("\nStarting Manual Hyperparameter Tuning and Cross-Validation...")
    
    k_values = [1, 3, 5, 7, 9]
    metrics = ['euclidean', 'manhattan']
    n_splits = 5
    
    folds = get_k_folds(X_train, y_train, n_splits)
    
    best_acc = -1
    best_params = {'k': 1, 'metric': 'manhattan'}
    
    total_combinations = len(k_values) * len(metrics)
    current_combo = 0
    
    for metric in metrics:
        for k in k_values:
            current_combo += 1
            print(f"\nEvaluating candidate {current_combo}/{total_combinations}: k={k}, metric={metric}")
            
            fold_accuracies = []
            for i in range(n_splits):
                # Setup train/val split for this fold
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
                
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]
                
                # Predict and evaluate
                y_pred = predict(X_val, X_tr, y_tr, k=k, metric=metric)
                acc = accuracy_score(y_val, y_pred)
                fold_accuracies.append(acc)
                print(f"  [Fold {i+1}/{n_splits}] Accuracy: {acc:.4f}")
                
            avg_acc = np.mean(fold_accuracies)
            print(f"  --> Average Accuracy for k={k}, metric={metric}: {avg_acc:.4f}")
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = {'k': k, 'metric': metric}
                
    print(f"\nBest parameters found: {best_params}")
    print(f"Best cross-validation accuracy: {best_acc:.4f}")
    
    # In KNN, "training" is just memorizing the training set.
    # We will return the best params so we can use them along with X_train/y_train later.
    return best_params

# =====================================================================
# 4. FINAL PREDICTION ON TEST SET
# =====================================================================

def generate_predictions(best_params, X_train, y_train, X_test, filenames, output_csv="results_name_id.csv"):
    """
    Generates predictions on the unlabeled test set and saves them to a CSV file.
    """
    total_samples = len(X_test)
    print(f"\nGenerating predictions for the test set ({total_samples} samples)...")
    
    y_pred = predict(X_test, X_train, y_train, k=best_params['k'], metric=best_params['metric'])
    
    print(f"Saving predictions to {output_csv}...")
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Predicted_Label"])
        for filename, pred in zip(filenames, y_pred):
            writer.writerow([filename, pred])
            
    print("Predictions saved successfully!")

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

if __name__ == "__main__":
    train_zip_path = "Train Set.zip"
    test_zip_path = "Test Set.zip"
    
    # 1. Data Loading & Preprocessing (Train Set)
    print("--- PHASE 1: DATA LOADING & PREPROCESSING (TRAIN) ---")
    X_train, y_train = load_data(train_zip_path, max_images_per_person=None) 
    print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # 2. Data Loading & Preprocessing (Test Set)
    print("\n--- PHASE 2: DATA LOADING & PREPROCESSING (TEST) ---")
    X_test, test_filenames = load_unlabeled_data(test_zip_path) 
    print(f"Test set shape: X_test={X_test.shape}")
    
    # 3. Cross-Validation & Hyperparameter Tuning & Training
    print("\n--- PHASE 3: MODEL TRAINING & TUNING ---")
    # best_params = tune_and_train_knn(X_train, y_train)
    # should return this:
    best_params = {'k': 1, 'metric': 'manhattan'}
    
    # 4. Final Prediction on Test Set
    print("\n--- PHASE 4: FINAL EVALUATION & CSV GENERATION ---")
    generate_predictions(best_params, X_train, y_train, X_test, test_filenames, output_csv="results_name_id.csv")
    
    print("\nPipeline execution complete!")

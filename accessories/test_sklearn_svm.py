import sys
import warnings
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main_structure"))

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from face_id.data import load_dataset
from face_id.splits import stratified_holdout_indices
from face_id.config import DATASET_PATH

def main():
    print("Loading a subset of data (max 5000 images)...", flush=True)
    t0 = time.time()
    X, y, label_to_name, _ = load_dataset(
        DATASET_PATH, 
        max_images=5000, 
        use_cache=True,
        verbose=False
    )
    print(f"Data loaded in {time.time() - t0:.1f}s", flush=True)
    
    train_idx, val_idx = stratified_holdout_indices(y, holdout_ratio=0.2, seed=42)
    X_train, y_train = X[train_idx], y[train_idx]
    
    print("Running PCA to 363 components...", flush=True)
    t0 = time.time()
    mean = X_train.mean(axis=0)
    _, S, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
    components = Vt[:363]
    
    Z_train = (X_train - mean) @ components.T
    print(f"PCA done in {time.time() - t0:.1f}s", flush=True)
    
    print("Scaling features...", flush=True)
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    
    setups = [
        {"C": 1.0, "max_iter": 2000, "dual": False, "scaled": False},
        {"C": 1.0, "max_iter": 2000, "dual": False, "scaled": True},
        {"C": 0.1, "max_iter": 2000, "dual": False, "scaled": True},
        {"C": 10.0, "max_iter": 5000, "dual": False, "scaled": True},
    ]
    
    print(f"\nTraining on {len(Z_train)} samples with 363 PCA features.\n", flush=True)
    
    for setup in setups:
        print(f"Testing setup: {setup}", flush=True)
        clf = LinearSVC(
            C=setup["C"], 
            max_iter=setup["max_iter"], 
            dual=setup["dual"], 
            multi_class="ovr", 
            random_state=42
        )
        
        X_to_use = Z_train_scaled if setup["scaled"] else Z_train
        
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(X_to_use, y_train)
            
            elapsed = time.time() - t0
            if len(w) > 0 and issubclass(w[-1].category, ConvergenceWarning):
                print(f"  -> FAILED to converge (Warning raised) in {elapsed:.1f}s", flush=True)
            else:
                print(f"  -> SUCCESS: Converged without warnings in {elapsed:.1f}s", flush=True)
        
        acc = clf.score(X_to_use, y_train)
        print(f"  -> Train accuracy: {acc:.4f}\n", flush=True)

if __name__ == "__main__":
    main()


"""Run the full supervised model evaluation on the entire dataset.

This script executes the Part B evaluation pipeline using all models
(including the new sklearn SVM), loads the full labeled dataset, and
performs hyperparameter search on an 80/20 train/validation split.

Run this script from the project root using:
    py -3.11 accessories/run_full_test.py
"""

import sys
import time
from pathlib import Path

# Add the main_structure directory to the path so we can import face_id
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main_structure"))

from face_id.part_b import run_part_b
from face_id.config import DATASET_PATH

def main():
    print("=" * 60)
    print(" STARTING FULL DATASET EVALUATION")
    print("=" * 60)
    print("This will test the following models:")
    print("  - k-NN")
    print("  - Linear least squares")
    print("  - Nearest class mean")
    print("  - Support Vector Machine (custom)")
    print("  - SVM (sklearn LinearSVC)")
    print("\nNote: Running on the full dataset (~16,000 images) with multiple")
    print("hyperparameters can take 20-45 minutes. You may see ConvergenceWarning")
    print("messages from LinearSVC; this is normal.\n")
    
    t0 = time.time()
    
    # Run the part B pipeline using all defaults from config, but we specify the 
    # models to explicitly include the new sklearn_svm
    models_to_run = ["knn", "linear_least_squares", "nearest_class_mean", "svm"]
    
    run_part_b(
        dataset_path=DATASET_PATH,
        model_names=models_to_run,
        use_cache=True, # Use cache to speed up subsequent runs
        verbose=True
    )
    
    elapsed = time.time() - t0
    print(f"\nEvaluation finished in {elapsed / 60:.1f} minutes.")

if __name__ == "__main__":
    main()

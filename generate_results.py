import argparse
import pandas as pd
from face_id.config import DATASET_PATH, DEFAULT_IMAGE_SIZE
from face_id.data import load_dataset
from face_id.model_registry import MODEL_SPECS
from face_id.pca import fit_pca, transform_pca

# TODO: Replace with your actual names and IDs
STUDENT_NAME = "YourName"
STUDENT_ID = "YourID"

def generate_csv(best_model_name, hyperparams, output_file=f"results_{STUDENT_NAME}_{STUDENT_ID}.csv"):
    print(f"Loading data for final submission...")
    # Load ALL images for the final training
    X, y, label_to_name, image_paths = load_dataset(DATASET_PATH, DEFAULT_IMAGE_SIZE, max_images_per_person=0)
    num_classes = len(label_to_name)
    
    print(f"Training final model ({best_model_name}) on ALL {len(X)} images...")
    model_spec = MODEL_SPECS[best_model_name]
    
    if model_spec.uses_pca:
        n_components = hyperparams.get("n_components", 50)
        pca_state, Z = fit_pca(X, n_components=n_components)
        model_state = model_spec.fit_fn(Z, y, num_classes, hyperparams)
        
        # Predicting on the same data just as a demonstration (Usually you predict on a Test set)
        print("Generating predictions...")
        y_pred, _ = model_spec.predict_fn(model_state, Z)
    else:
        model_state = model_spec.fit_fn(X, y, num_classes, hyperparams)
        print("Generating predictions...")
        y_pred, _ = model_spec.predict_fn(model_state, X)
    
    # Create the DataFrame
    results_df = pd.DataFrame({
        "Image_Path": image_paths,
        "True_Label": [label_to_name[label] for label in y],
        "Predicted_Label": [label_to_name[label] for label in y_pred]
    })
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Successfully saved predictions to {output_file}!")

if __name__ == "__main__":
    # We will update these values once run_part_b finishes!
    BEST_MODEL = "knn" 
    BEST_HYPERPARAMS = {"k": 1, "n_components": 363}
    
    generate_csv(BEST_MODEL, BEST_HYPERPARAMS)

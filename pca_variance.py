import matplotlib.pyplot as plt
import numpy as np
from face_id.data import load_dataset
from face_id.config import DATASET_PATH, DEFAULT_IMAGE_SIZE
from face_id.pca import fit_pca

def plot_explained_variance():
    print("Loading data for PCA variance analysis...")
    X, y, label_to_name, _ = load_dataset(DATASET_PATH, DEFAULT_IMAGE_SIZE, max_images=None, max_images_per_person=200, use_cache=True)
    
    print(f"Loaded {len(X)} images. Computing full PCA...")
    # Calculate PCA with a large number of components to see the curve
    max_comp = min(400, X.shape[0], X.shape[1])
    pca_state, _ = fit_pca(X, n_components=max_comp)
    
    explained_variance = pca_state.explained_variance
    # Convert to ratio by dividing by sum of all eigenvalues (approximate by sum of these top ones, or better: sum of variances of X)
    total_variance = np.var(X, axis=0).sum()
    explained_variance_ratio = explained_variance / total_variance
    
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find the number of components for 95% variance
    target_variance = 0.95
    components_95 = np.argmax(cumulative_variance >= target_variance) + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='-')
    plt.axhline(y=target_variance, color='r', linestyle='--', label='95% Variance')
    plt.axvline(x=components_95, color='g', linestyle='--', label=f'{components_95} Components')
    
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca_variance.png')
    print(f"\nResults:")
    print(f"To keep 95% of the information, you need EXACTLY {components_95} PCA components.")
    plt.show()

if __name__ == "__main__":
    plot_explained_variance()

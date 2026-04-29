import matplotlib.pyplot as plt
import numpy as np
from face_id.data import load_dataset
from face_id.config import DATASET_PATH

def show_random_samples(n=10):
    print("Loading dataset for visualization...")
    X, y, label_to_name, image_paths = load_dataset(DATASET_PATH, max_images=500, verbose=False)
    
    indices = np.random.choice(len(X), n, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Reshape the flattened vector back to 2D (assuming 64x72 based on Exercise 2 PDF)
        # Note: Original is 64x72, but we'll check the actual shape from the data
        img_dim = int(np.sqrt(X.shape[1])) # Fallback if not square
        # But we know from PDF it's 64x72. Let's try to infer or use standard.
        try:
            img = X[idx].reshape(64, 72)
        except:
            # Fallback to square if 64x72 fails
            side = int(np.sqrt(X.shape[1]))
            img = X[idx].reshape(side, side)
            
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(label_to_name[y[idx]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_random_samples(10)

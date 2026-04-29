import argparse
import matplotlib

matplotlib.use('TkAgg')  # Fix for PyCharm backend issue
import matplotlib.pyplot as plt
import numpy as np

from face_id.config import DATASET_PATH, DEFAULT_IMAGE_SIZE
from face_id.data import load_dataset
from face_id.metrics import accuracy_score
from face_id.models import knn, linear_least_squares
from face_id.pca import fit_pca, transform_pca
from face_id.splits import train_test_split


def show_sample_images(X, y, label_to_name, count=10, image_size=DEFAULT_IMAGE_SIZE):
    """Display a grid of sample face images from the flattened dataset."""
    count = min(count, len(X))
    if count == 0:
        print("No images available to display.")
        return

    image_shape = (64, 72) if image_size is None else tuple(image_size)
    cols = 5
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows))
    axes = np.asarray(axes).reshape(-1)

    for index in range(count):
        image = X[index].reshape(image_shape)
        axes[index].imshow(image, cmap="gray")
        axes[index].set_title(label_to_name[int(y[index])])
        axes[index].axis("off")

    for axis in axes[count:]:
        axis.axis("off")

    fig.suptitle(f"First {count} dataset images")
    plt.tight_layout()
    plt.show()


def plot_true_bias_variance(param_values, train_accs, val_accs, param_name, model_name, invert_x=False):
    """
    Plots the classic Bias-Variance tradeoff graph using Error rates.
    Bias proxy = Training Error
    Variance proxy = Validation Error - Training Error
    Total Error = Validation Error
    """
    # Convert Accuracy to Error Rate
    train_errors = [1.0 - acc for acc in train_accs]
    val_errors = [1.0 - acc for acc in val_accs]

    # Calculate proxies for Bias and Variance
    bias = train_errors
    variance = [max(0, val - train) for val, train in zip(val_errors, train_errors)]
    total_error = val_errors

    plt.figure(figsize=(9, 6))

    # Plot the three theoretical lines
    plt.plot(param_values, bias, label='Bias Proxy (Train Error)',
             linestyle='--', marker='o', color='blue', linewidth=2)

    plt.plot(param_values, variance, label='Variance Proxy (Val-Train Gap)',
             linestyle='-.', marker='^', color='orange', linewidth=2)

    plt.plot(param_values, total_error, label='Total Error (Val Error)',
             linestyle='-', marker='s', color='red', linewidth=3)

    # Formatting the plot
    plt.title(f'{model_name}: Bias-Variance Tradeoff')
    plt.ylabel('Error Rate (Lower is Better)')

    if invert_x:
        plt.gca().invert_xaxis()
        plt.xlabel(f'{param_name}\n<-- High Complexity (Overfitting)  |  Low Complexity (Underfitting) -->')
    else:
        plt.xlabel(f'{param_name}\n<-- Low Complexity (Underfitting)  |  High Complexity (Overfitting) -->')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot as a PNG file in the current directory
    filename = f"{model_name.replace(' ', '_').replace('+', 'and').lower()}_error_curve.png"
    plt.savefig(filename)
    print(f"  -> Saved graph as: {filename}")

    # Display the plot window
    plt.show()


def evaluate_knn(X_train, X_val, y_train, y_val):
    print("\nEvaluating KNN...")
    k_values = [1, 3, 5, 10, 20, 50]
    train_accs = []
    val_accs = []

    for k in k_values:
        model_state = knn.fit(X_train, y_train, num_classes=int(np.max(y_train)) + 1, hyperparams={"k": k})
        y_train_pred, _ = knn.predict(model_state, X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        y_val_pred, _ = knn.predict(model_state, X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"  k={k:2d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # For KNN, small 'k' means higher complexity, so we invert the X axis
    plot_true_bias_variance(k_values, train_accs, val_accs, 'k (Number of Neighbors)', 'KNN', invert_x=False)


def evaluate_pca_ls(X_train, X_val, y_train, y_val, num_classes=28):
    print("\nEvaluating PCA + Least Squares...")
    components_list = [2, 5, 10, 20, 50, 100, 200]
    train_accs = []
    val_accs = []

    for n in components_list:
        # Fit PCA and Linear Model on Training data
        pca_state, Z_train = fit_pca(X_train, n_components=n)
        model_state = linear_least_squares.fit(Z_train, y_train, num_classes, hyperparams={})

        # Predict on Training data
        y_train_pred, _ = linear_least_squares.predict(model_state, Z_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Predict on Validation data
        Z_val = transform_pca(X_val, pca_state)
        y_val_pred, _ = linear_least_squares.predict(model_state, Z_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"  Components={n:3d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # For PCA, more components mean higher complexity
    plot_true_bias_variance(components_list, train_accs, val_accs, 'Number of PCA Components', 'PCA + Least Squares',
                            invert_x=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show sample face images and optionally plot model graphs.")
    parser.add_argument("--count", type=int, default=10, help="Number of sample images to show.")
    parser.add_argument("--graphs", action="store_true", help="Also run the bias-variance graph evaluations.")
    args = parser.parse_args()

    print("Loading and preparing data...")
    max_images = None if args.graphs else args.count
    X, y, label_to_name, image_paths = load_dataset(
        DATASET_PATH,
        DEFAULT_IMAGE_SIZE,
        max_images=max_images,
        cache_name="graphs",
    )
    show_sample_images(X, y, label_to_name, count=args.count, image_size=DEFAULT_IMAGE_SIZE)

    if not args.graphs:
        print("\nDisplayed sample images successfully.")
        raise SystemExit(0)

    # Split into Train and Validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_ratio=0.2, seed=42)
    print(f"Data ready. Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    # Run evaluations and plot graphs
    evaluate_knn(X_train, X_val, y_train, y_val)
    evaluate_pca_ls(X_train, X_val, y_train, y_val, num_classes=28)

    print("\nExecution completed successfully!")

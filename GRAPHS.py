import matplotlib

matplotlib.use('TkAgg')  # Fix for PyCharm backend issue
import matplotlib.pyplot as plt
import numpy as np

from knn import *
from linear_model import *

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
        y_train_pred = predict(X_train, X_train, y_train, k=k)
        train_acc = accuracy(y_train, y_train_pred)

        y_val_pred = predict(X_val, X_train, y_train, k=k)
        val_acc = accuracy(y_val, y_val_pred)

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
        mean_face, components, explained_var, Z_train = fit_pca(X_train, n_components=n)
        W = fit_linear_least_squares(Z_train, y_train, num_classes)

        # Predict on Training data
        y_train_pred, _ = predict_linear(Z_train, W)
        train_acc = accuracy_score_numpy(y_train, y_train_pred)

        # Predict on Validation data
        Z_val = transform_pca(X_val, mean_face, components)
        y_val_pred, _ = predict_linear(Z_val, W)
        val_acc = accuracy_score_numpy(y_val, y_val_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"  Components={n:3d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # For PCA, more components mean higher complexity
    plot_true_bias_variance(components_list, train_accs, val_accs, 'Number of PCA Components', 'PCA + Least Squares',
                            invert_x=False)


if __name__ == "__main__":
    print("Loading and preparing data...")
    # Adjust according to how your actual load_dataset function returns values
    X, y, label_to_name, image_paths = load_dataset(DATASET_PATH, IMAGE_SIZE)

    # Split into Train and Validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_ratio=0.2, seed=42)
    print(f"Data ready. Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    # Run evaluations and plot graphs
    evaluate_knn(X_train, X_val, y_train, y_val)
    evaluate_pca_ls(X_train, X_val, y_train, y_val, num_classes=28)

    print("\nExecution completed successfully!")
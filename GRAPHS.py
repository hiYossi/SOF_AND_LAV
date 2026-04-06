import numpy as np
import matplotlib.pyplot as plt




def plot_knn_validation_curve(X_train, X_val, y_train, y_val):
    print("Generating KNN Validation Curve...")
    k_values = [1, 3, 5, 10, 20, 50]
    train_accs = []
    val_accs = []

    for k in k_values:
        # דיוק אימון (לבדוק Overfitting)
        y_train_pred = predict(X_train, X_train, y_train, k=k)
        train_acc = accuracy(y_train, y_train_pred)

        # דיוק ולידציה (לבדוק הכללה)
        y_val_pred = predict(X_val, X_train, y_train, k=k)
        val_acc = accuracy(y_val, y_val_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"  KNN k={k}: Train={train_acc:.3f}, Val={val_acc:.3f}")

    # ציור הגרף
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_accs, label='Train Accuracy', marker='o', linewidth=2)
    plt.plot(k_values, val_accs, label='Validation Accuracy', marker='s', linewidth=2)
    plt.title('KNN: Bias-Variance Tradeoff')
    plt.xlabel('k (Number of Neighbors) - Decreasing Complexity ->')
    plt.ylabel('Accuracy')
    plt.gca().invert_xaxis()  # הופכים את הציר: k קטן (מורכב) משמאל, k גדול (פשוט) מימין
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('knn_bias_variance.png')  # שומר את הגרף כתמונה
    plt.show()


def plot_pca_ls_validation_curve(X_train, X_val, y_train, y_val, num_classes=28):
    print("\nGenerating PCA+LS Validation Curve...")
    components_list = [2, 5, 10, 20, 50, 100, 200]
    train_accs = []
    val_accs = []

    for n in components_list:
        # אימון מודל ה-PCA והמסווג
        mean_face, components, explained_var, Z_train = fit_pca(X_train, n_components=n)
        W = fit_linear_least_squares(Z_train, y_train, num_classes)

        # דיוק אימון
        y_train_pred, _ = predict_linear(Z_train, W)
        train_acc = accuracy_score_numpy(y_train, y_train_pred)

        # דיוק ולידציה
        Z_val = transform_pca(X_val, mean_face, components)
        y_val_pred, _ = predict_linear(Z_val, W)
        val_acc = accuracy_score_numpy(y_val, y_val_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"  PCA n={n}: Train={train_acc:.3f}, Val={val_acc:.3f}")

    # ציור הגרף
    plt.figure(figsize=(8, 5))
    plt.plot(components_list, train_accs, label='Train Accuracy', marker='o', linewidth=2)
    plt.plot(components_list, val_accs, label='Validation Accuracy', marker='s', linewidth=2)
    plt.title('PCA + Least Squares: Bias-Variance Tradeoff')
    plt.xlabel('Number of PCA Components (Increasing Complexity ->)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('pca_ls_bias_variance.png')  # שומר את הגרף כתמונה
    plt.show()


if __name__ == "__main__":
    # 1. טעינת הנתונים (השתמש בפונקציה מהקובץ שלך. שמתי כאן את הגרסה מהקוד השני שלך)
    print("Loading data...")
    # בהתאם לאיך שקראת לפונקציה בקובץ המקורי שלך:
    X, y, label_to_name, image_paths = load_dataset(DATASET_PATH, IMAGE_SIZE)

    # 2. חלוקה ל-Train ו-Validation
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_ratio=0.2, seed=42)

    # 3. יצירת הגרפים (שומר אותם כקובצי PNG בתיקייה ומציג על המסך)
    plot_knn_validation_curve(X_train, X_val, y_train, y_val)
    plot_pca_ls_validation_curve(X_train, X_val, y_train, y_val, num_classes=28)

    print("\nDone! Graphs saved to your folder.")
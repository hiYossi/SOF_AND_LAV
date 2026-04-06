"""
PyTorch face recognition training script.

This implementation keeps the original image resolution and uses all available
images by default. The training pipeline includes:

- stratified train/validation/test splitting
- light data augmentation on the training set
- a stronger CNN with BatchNorm and dropout
- validation-based checkpointing
- AdamW + ReduceLROnPlateau + early stopping
"""

import argparse
import time
import re
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = None


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "Train Set (Labeled)-20260405T164823Z-3-001.zip"
IMAGE_SIZE = None
MAX_IMAGES_PER_PERSON = 0
VAL_RATIO = 0.1
TEST_RATIO = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6


# ============================================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================================

def require_torch():
    """Fail with a helpful message when PyTorch is not installed."""
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for training this model. "
            "Install it with `pip install torch` and rerun the script."
        )


def seed_everything(seed):
    """Set random seeds for reproducible experiments."""
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_pgm_from_stream(stream):
    """
    Load a binary P5 PGM image from an open stream.
    """
    magic = stream.readline().strip()
    if magic != b"P5":
        raise ValueError("Not a valid P5 PGM file")

    while True:
        position = stream.tell()
        line = stream.readline().strip()
        if not line.startswith(b"#"):
            stream.seek(position)
            break

    width, height = map(int, stream.readline().strip().split())
    max_value = int(stream.readline().strip())

    dtype = np.uint8 if max_value < 256 else np.uint16
    data = np.frombuffer(stream.read(), dtype=dtype)
    return data.reshape((height, width))


def preprocess_image(image, target_size=None):
    """Normalize to [0, 1], preserving original resolution by default."""
    if target_size is None:
        resized = image.astype(np.float32)
    else:
        target_height, target_width = target_size
        height, width = image.shape

        rows = np.floor(np.arange(target_height) * height / target_height).astype(int)
        cols = np.floor(np.arange(target_width) * width / target_width).astype(int)
        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)
        resized = image[np.ix_(rows, cols)].astype(np.float32)

    min_value = resized.min()
    max_value = resized.max()
    if max_value > min_value:
        resized = (resized - min_value) / (max_value - min_value)
    else:
        resized = np.zeros_like(resized, dtype=np.float32)

    return resized


# ============================================================================
# DATASET LOADING
# ============================================================================

def collect_balanced_members(zip_name, max_images_per_person, seed):
    """
    Collect image members from the zip archive.

    Returns:
        members_with_labels: list of (internal_zip_path, class_index)
        label_to_name: mapping of class index to readable label name
    """
    rng = np.random.default_rng(seed)
    members_by_person = defaultdict(list)

    with zipfile.ZipFile(zip_name, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".pgm"):
                continue
            match = re.search(r"(?:^|/)p(\d+)_i\d+\.pgm$", member)
            if match:
                members_by_person[int(match.group(1))].append(member)

    if not members_by_person:
        raise ValueError("No labeled PGM files were found in {}".format(zip_name))

    person_ids = sorted(members_by_person)
    label_to_name = dict(
        (index, "person_{}".format(person_id))
        for index, person_id in enumerate(person_ids)
    )
    person_to_label = dict(
        (person_id, index) for index, person_id in enumerate(person_ids)
    )

    selected_members = []
    for person_id in person_ids:
        person_members = members_by_person[person_id]
        permutation = rng.permutation(len(person_members))
        if max_images_per_person is None:
            limit = len(person_members)
        else:
            limit = min(max_images_per_person, len(person_members))
        for idx in permutation[:limit]:
            selected_members.append((person_members[idx], person_to_label[person_id]))

    rng.shuffle(selected_members)
    return selected_members, label_to_name


def load_dataset(zip_name, image_size=None, max_images_per_person=None,
                 seed=42, verbose=True):
    """
    Load the dataset from the project zip file.

    Returns:
        X: image tensor data as numpy array, shape (n_samples, 1, H, W)
        y: class labels, shape (n_samples,)
        label_to_name: mapping from class index to label
        image_paths: original zip member paths
    """
    selected_members, label_to_name = collect_balanced_members(
        zip_name, max_images_per_person=max_images_per_person, seed=seed
    )

    X_list = []
    y_list = []
    image_paths = []

    with zipfile.ZipFile(zip_name, "r") as zf:
        for index, (member, class_label) in enumerate(selected_members, start=1):
            with zf.open(member) as image_file:
                image = load_pgm_from_stream(image_file)
            processed = preprocess_image(image, target_size=image_size)
            X_list.append(processed)
            y_list.append(class_label)
            image_paths.append(member)

            if verbose and index % 100 == 0:
                print("  Loaded {} images...".format(index))

    X = np.stack(X_list, axis=0).astype(np.float32)
    X = X[:, np.newaxis, :, :]
    y = np.array(y_list, dtype=np.int64)
    return X, y, label_to_name, image_paths


def split_counts_for_class(num_items, val_ratio, test_ratio):
    """Compute stable per-class split sizes while keeping at least one train item."""
    if num_items <= 1:
        return 0, 0

    test_count = int(round(num_items * test_ratio)) if test_ratio > 0 else 0
    val_count = int(round(num_items * val_ratio)) if val_ratio > 0 else 0

    if test_ratio > 0 and test_count == 0:
        test_count = 1

    if val_ratio > 0 and num_items >= 3 and val_count == 0:
        val_count = 1

    if num_items < 3:
        val_count = 0

    while test_count + val_count > num_items - 1:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    return val_count, test_count


def stratified_split_indices(y, val_ratio, test_ratio, seed=42):
    """Create stratified train/validation/test index splits."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1")

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        class_indices = rng.permutation(class_indices)

        val_count, test_count = split_counts_for_class(
            len(class_indices), val_ratio=val_ratio, test_ratio=test_ratio
        )

        test_indices.extend(class_indices[:test_count].tolist())
        val_start = test_count
        val_end = test_count + val_count
        val_indices.extend(class_indices[val_start:val_end].tolist())
        train_indices.extend(class_indices[val_end:].tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def compute_normalization_stats(X, indices):
    """Compute mean/std from the training split only."""
    selected = X[indices]
    mean = float(selected.mean())
    std = float(selected.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


# ============================================================================
# AUGMENTATION AND DATASETS
# ============================================================================

def translate_tensor(image, shift_y, shift_x):
    """Translate an image with zero fill instead of wraparound."""
    shifted = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
    if shift_y > 0:
        shifted[:, :shift_y, :] = 0
    elif shift_y < 0:
        shifted[:, shift_y:, :] = 0

    if shift_x > 0:
        shifted[:, :, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, :, shift_x:] = 0

    return shifted


def augment_face_tensor(image):
    """Apply light geometric and photometric augmentation."""
    image = image.clone()

    if torch.rand(1).item() < 0.9:
        shift_y = int(torch.randint(-3, 4, (1,)).item())
        shift_x = int(torch.randint(-3, 4, (1,)).item())
        image = translate_tensor(image, shift_y, shift_x)

    contrast = float(torch.empty(1).uniform_(0.90, 1.10).item())
    brightness = float(torch.empty(1).uniform_(-0.08, 0.08).item())
    image_mean = image.mean()
    image = (image - image_mean) * contrast + image_mean + brightness

    if torch.rand(1).item() < 0.5:
        noise_scale = float(torch.empty(1).uniform_(0.0, 0.025).item())
        image = image + torch.randn_like(image) * noise_scale

    return torch.clamp(image, 0.0, 1.0)


if Dataset is None:
    class FaceDataset(object):
        """Placeholder that raises a helpful error when PyTorch is missing."""

        def __init__(self, *args, **kwargs):
            require_torch()
else:
    class FaceDataset(Dataset):
        """Dataset wrapper with optional augmentation and train-set normalization."""

        def __init__(self, images, labels, indices, mean, std, augment=False):
            self.images = images
            self.labels = labels
            self.indices = np.array(indices, dtype=np.int64)
            self.mean = float(mean)
            self.std = float(std)
            self.augment = augment

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            index = int(self.indices[item])
            image = torch.from_numpy(self.images[index]).clone()
            label = int(self.labels[index])

            if self.augment:
                image = augment_face_tensor(image)

            image = (image - self.mean) / self.std
            return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(X, y, train_indices, val_indices, test_indices, batch_size,
                       augment_train=True):
    """Build dataloaders from indexed splits."""
    require_torch()

    train_mean, train_std = compute_normalization_stats(X, train_indices)

    train_dataset = FaceDataset(
        X, y, train_indices, mean=train_mean, std=train_std, augment=augment_train
    )
    train_eval_dataset = FaceDataset(
        X, y, train_indices, mean=train_mean, std=train_std, augment=False
    )
    val_dataset = FaceDataset(
        X, y, val_indices, mean=train_mean, std=train_std, augment=False
    )
    test_dataset = FaceDataset(
        X, y, test_indices, mean=train_mean, std=train_std, augment=False
    )

    common_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    train_eval_loader = DataLoader(
        train_eval_dataset, shuffle=False, **common_loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    return train_loader, train_eval_loader, val_loader, test_loader, train_mean, train_std


# ============================================================================
# MODEL
# ============================================================================

if nn is None:
    class BetterFaceCNN(object):
        """Placeholder that raises a helpful error when PyTorch is missing."""

        def __init__(self, *args, **kwargs):
            require_torch()
else:
    class ConvBlock(nn.Module):
        """Two-layer convolutional block with BatchNorm."""

        def __init__(self, in_channels, out_channels, dropout):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(p=dropout),
            )

        def forward(self, x):
            return self.block(x)


    class BetterFaceCNN(nn.Module):
        """Stronger CNN for grayscale face classification."""

        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(1, 32, dropout=0.05),
                ConvBlock(32, 64, dropout=0.08),
                ConvBlock(64, 128, dropout=0.12),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.35),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)


# ============================================================================
# TRAINING
# ============================================================================

def evaluate(model, loader, criterion, device):
    """Evaluate loss and accuracy on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += inputs.size(0)

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return mean_loss, accuracy


def train_model(model, train_loader, val_loader, epochs, learning_rate,
                weight_decay, patience, device):
    """Train the CNN using validation performance for model selection."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_state_dict = None
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_seen = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += batch_size

        train_loss = running_loss / max(total_seen, 1)
        train_accuracy = running_correct / max(total_seen, 1)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "  Epoch {:02d}/{} | lr={:.6f} | train_loss={:.4f} | "
            "train_acc={:.3f} | val_loss={:.4f} | val_acc={:.3f}".format(
                epoch,
                epochs,
                current_lr,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )
        )

        improved = (
            val_accuracy > best_val_accuracy or
            (val_accuracy == best_val_accuracy and val_loss < best_val_loss)
        )

        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = dict(
                (key, value.detach().cpu().clone())
                for key, value in model.state_dict().items()
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered after epoch {}.".format(epoch))
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def show_sample_predictions(model, X, y, test_indices, label_to_name, image_paths,
                            mean, std, device, limit=5):
    """Print a few example predictions from the test set."""
    print("\n[7] Sample predictions...")

    selected_indices = test_indices[:limit]
    raw_images = torch.from_numpy(X[selected_indices].copy())
    raw_images = (raw_images - mean) / std
    raw_images = raw_images.to(device)

    with torch.no_grad():
        logits = model(raw_images)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)

    for row, predicted_label in enumerate(predictions):
        global_index = int(selected_indices[row])
        true_label = int(y[global_index])
        confidence = float(probabilities[row, predicted_label])
        marker = "OK" if predicted_label == true_label else "MISS"
        print(
            "  {:4} {} | true={} | pred={} | conf={:.3f}".format(
                marker,
                image_paths[global_index],
                label_to_name[true_label],
                label_to_name[predicted_label],
                confidence,
            )
        )


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train an improved PyTorch CNN for face recognition."
    )
    parser.add_argument("--dataset", default=DATASET_PATH,
                        help="Path to the labeled zip dataset.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Optional square image size after resizing. Omit to keep original resolution."
    )
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=MAX_IMAGES_PER_PERSON,
        help="Balanced sampling limit per identity. Use 0 for all images."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Fraction of each class reserved for validation."
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help="Fraction of each class reserved for testing."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Maximum training epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay for AdamW.")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help="Early stopping patience based on validation improvement.")
    parser.add_argument("--disable-augmentation", action="store_true",
                        help="Disable training-set augmentation.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only load and split the dataset without training.")
    return parser


def main():
    args = build_parser().parse_args()
    image_size = None if args.image_size is None else (args.image_size, args.image_size)
    max_images_per_person = None if args.max_images_per_person <= 0 else args.max_images_per_person

    seed_everything(args.seed)

    print("=" * 70)
    print("FACE RECOGNITION: IMPROVED PYTORCH CNN")
    print("=" * 70)

    print("\n[1] Loading dataset...")
    load_start = time.perf_counter()
    X, y, label_to_name, image_paths = load_dataset(
        args.dataset,
        image_size=image_size,
        max_images_per_person=max_images_per_person,
        seed=args.seed,
    )
    load_seconds = time.perf_counter() - load_start

    print("  Loaded samples: {}".format(len(X)))
    print("  Classes: {}".format(len(label_to_name)))
    print("  Image tensor shape: {}".format(X.shape[1:]))
    print("  Image size mode: {}".format(
        "original resolution" if image_size is None else image_size
    ))
    print("  Max images per person: {}".format(max_images_per_person or "ALL"))
    print("  Load time: {:.1f}s".format(load_seconds))

    print("\n[2] Creating stratified train/validation/test split...")
    train_indices, val_indices, test_indices = stratified_split_indices(
        y, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    print("  Train samples: {}".format(len(train_indices)))
    print("  Validation samples: {}".format(len(val_indices)))
    print("  Test samples: {}".format(len(test_indices)))
    print("  First train path: {}".format(image_paths[int(train_indices[0])]))

    if args.dry_run:
        print("\nDry run complete. Install PyTorch and rerun without --dry-run to train.")
        return

    require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[3] Using device: {}".format(device))

    train_loader, train_eval_loader, val_loader, test_loader, train_mean, train_std = (
        create_dataloaders(
            X,
            y,
            train_indices,
            val_indices,
            test_indices,
            batch_size=args.batch_size,
            augment_train=not args.disable_augmentation,
        )
    )
    print("  Train normalization mean/std: {:.4f} / {:.4f}".format(
        train_mean, train_std
    ))

    print("\n[4] Building model...")
    model = BetterFaceCNN(num_classes=len(label_to_name)).to(device)
    print(model)

    print("\n[5] Training...")
    train_start = time.perf_counter()
    training_summary = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )
    train_seconds = time.perf_counter() - train_start
    model = training_summary["model"]

    criterion = nn.CrossEntropyLoss()
    train_loss, train_accuracy = evaluate(model, train_eval_loader, criterion, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    print("\n[6] Final metrics...")
    print("  Train loss: {:.4f}".format(train_loss))
    print("  Train accuracy: {:.3f}".format(train_accuracy))
    print("  Validation loss: {:.4f}".format(val_loss))
    print("  Validation accuracy: {:.3f}".format(val_accuracy))
    print("  Test loss: {:.4f}".format(test_loss))
    print("  Test accuracy: {:.3f}".format(test_accuracy))
    print("  Best validation accuracy: {:.3f} (epoch {})".format(
        training_summary["best_val_accuracy"],
        training_summary["best_epoch"],
    ))
    print("  Training time: {:.1f}s".format(train_seconds))

    show_sample_predictions(
        model,
        X,
        y,
        test_indices,
        label_to_name,
        image_paths,
        mean=train_mean,
        std=train_std,
        device=device,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

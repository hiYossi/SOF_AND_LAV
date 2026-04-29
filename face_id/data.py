"""Dataset loading and preprocessing helpers."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, RANDOM_SEED


PGM_PATTERN = re.compile(r"(?:.*/)?p(\d+)_i(\d+)(?:\(\d+\))?\.pgm$")


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


def load_pgm(path_like):
    """Load a PGM image from disk or from a zip-member path."""
    path_like = str(path_like)
    if ".zip" in path_like:
        zip_path, internal_path = path_like.split(".zip", 1)
        zip_path += ".zip"
        internal_path = internal_path.lstrip("/\\")
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(internal_path) as file_obj:
                return _read_pgm(file_obj)

    with open(path_like, "rb") as file_obj:
        return _read_pgm(file_obj)


def preprocess_image(image, target_size=None):
    """Detect gray bars, crop to 64x64, resize optionally, and normalize."""
    # 1. Detect and crop gray bars (from 64x72 to 64x64)
    if image.shape == (64, 72):
        left_side = image[:, :8]
        right_side = image[:, -8:]
        
        # The gray bar usually has very low variance (uniform color)
        if np.var(left_side) < np.var(right_side):
            # Gray bar is on the left
            image = image[:, 8:]
        else:
            # Gray bar is on the right
            image = image[:, :64]

    # 2. Resizing logic
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

    # 3. Normalization logic (Z-score standardization)
    mean_val = float(resized.mean())
    std_val = float(resized.std())
    if std_val > 0:
        resized = (resized - mean_val) / std_val
    else:
        resized = resized - mean_val


    return resized.astype(np.float32)


def parse_image_size(image_height, image_width):
    """Convert two optional CLI dimensions into an image-size tuple."""
    if image_height is None and image_width is None:
        return None
    if image_height is None or image_width is None:
        raise ValueError("Provide both image_height and image_width, or neither.")
    return int(image_height), int(image_width)


def parse_max_images_per_person(value):
    """Treat non-positive values as 'use all images per person'."""
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


def _dataset_cache_key(dataset_path, image_size, max_images, max_images_per_person,
                       random_seed):
    dataset_path = Path(dataset_path)
    stat = dataset_path.stat()
    size_key = "original" if image_size is None else f"{image_size[0]}x{image_size[1]}"
    return (
        f"{dataset_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{size_key}|"
        f"{max_images}|{max_images_per_person}|{random_seed}"
    )


def _dataset_cache_path(cache_name, cache_key):
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"{cache_name}_{digest}.npz"


def _load_cached_dataset(cache_path):
    with np.load(cache_path, allow_pickle=False) as data:
        label_names = [str(name) for name in data["label_names"].tolist()]
        label_to_name = {idx: name for idx, name in enumerate(label_names)}
        image_paths = [str(path) for path in data["image_paths"].tolist()]
        return data["X"], data["y"].astype(np.int32), label_to_name, image_paths


def _save_cached_dataset(cache_path, X, y, label_to_name, image_paths):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    label_names = np.array([label_to_name[idx] for idx in sorted(label_to_name)])
    np.savez(
        cache_path,
        X=X,
        y=y,
        label_names=label_names,
        image_paths=np.array(image_paths),
    )


def _collect_zip_records(dataset_path):
    grouped_records = {}
    with zipfile.ZipFile(dataset_path, "r") as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".pgm")]

    if not members:
        raise ValueError(f"No PGM files found in {dataset_path}.")

    for member in members:
        match = PGM_PATTERN.match(member)
        if not match:
            continue
        person_id = int(match.group(1))
        grouped_records.setdefault(person_id, []).append(member)

    if not grouped_records:
        raise ValueError(
            "No labeled PGM files matched the expected p{person}_i{image}.pgm pattern."
        )

    return grouped_records


def _collect_directory_records(dataset_path):
    grouped_records = {}
    for class_dir in sorted(Path(dataset_path).iterdir()):
        if not class_dir.is_dir():
            continue
        image_paths = sorted(class_dir.glob("*.pgm"))
        if image_paths:
            grouped_records[class_dir.name] = image_paths

    if not grouped_records:
        raise ValueError(f"No class folders with PGM files found in {dataset_path}.")

    return grouped_records


def _sample_grouped_records(grouped_records, max_images, max_images_per_person, random_seed):
    rng = np.random.default_rng(random_seed)
    selected = []

    for raw_label in sorted(grouped_records):
        records = list(grouped_records[raw_label])
        if max_images_per_person is not None:
            permutation = rng.permutation(len(records))
            keep_count = min(max_images_per_person, len(records))
            records = [records[idx] for idx in permutation[:keep_count]]
        selected.extend((record, raw_label) for record in records)

    if max_images is not None and len(selected) > max_images:
        permutation = rng.permutation(len(selected))
        selected = [selected[idx] for idx in permutation[:max_images]]

    return selected


def load_dataset(dataset_path, image_size=None, max_images=None,
                 max_images_per_person=None, cache_name="dataset",
                 use_cache=True, random_seed=RANDOM_SEED, verbose=True):
    """
    Load and preprocess the face dataset.

    Returns:
        X (ndarray): Flattened feature matrix.
        y (ndarray): Integer labels from 0..C-1.
        label_to_name (dict): Mapping from class index to person name.
        image_paths (list[str]): Original paths or zip-member paths.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    max_images_per_person = parse_max_images_per_person(max_images_per_person)
    cache_key = _dataset_cache_key(
        dataset_path,
        image_size=image_size,
        max_images=max_images,
        max_images_per_person=max_images_per_person,
        random_seed=random_seed,
    )
    cache_path = _dataset_cache_path(cache_name, cache_key)

    if use_cache and cache_path.exists():
        if verbose:
            print(f"  Loaded cached dataset from {cache_path.name}")
        return _load_cached_dataset(cache_path)

    if dataset_path.suffix.lower() == ".zip":
        grouped_records = _collect_zip_records(dataset_path)
        selected = _sample_grouped_records(
            grouped_records,
            max_images=max_images,
            max_images_per_person=max_images_per_person,
            random_seed=random_seed,
        )
        raw_labels = sorted(grouped_records)
        raw_to_idx = {raw_label: idx for idx, raw_label in enumerate(raw_labels)}
        label_to_name = {idx: f"person_{raw_label}" for raw_label, idx in raw_to_idx.items()}

        X_list = []
        y_list = []
        image_paths = []

        with zipfile.ZipFile(dataset_path, "r") as archive:
            for index, (member, raw_label) in enumerate(selected, start=1):
                with archive.open(member) as file_obj:
                    image = _read_pgm(file_obj)
                image = preprocess_image(image, target_size=image_size)
                X_list.append(image.flatten())
                y_list.append(raw_to_idx[raw_label])
                image_paths.append(f"{dataset_path}/{member}")
                if verbose and index % 200 == 0:
                    print(f"  Loaded {index} images...")
    else:
        grouped_records = _collect_directory_records(dataset_path)
        selected = _sample_grouped_records(
            grouped_records,
            max_images=max_images,
            max_images_per_person=max_images_per_person,
            random_seed=random_seed,
        )
        raw_labels = sorted(grouped_records)
        raw_to_idx = {raw_label: idx for idx, raw_label in enumerate(raw_labels)}
        label_to_name = {idx: str(raw_label) for raw_label, idx in raw_to_idx.items()}

        X_list = []
        y_list = []
        image_paths = []

        for index, (path, raw_label) in enumerate(selected, start=1):
            image = preprocess_image(load_pgm(path), target_size=image_size)
            X_list.append(image.flatten())
            y_list.append(raw_to_idx[raw_label])
            image_paths.append(str(path))
            if verbose and index % 200 == 0:
                print(f"  Loaded {index} images...")

    if not X_list:
        raise ValueError("No images were loaded after applying the current limits.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)

    if use_cache:
        _save_cached_dataset(cache_path, X, y, label_to_name, image_paths)
        if verbose:
            print(f"  Saved dataset cache to {cache_path.name}")

    return X, y, label_to_name, image_paths

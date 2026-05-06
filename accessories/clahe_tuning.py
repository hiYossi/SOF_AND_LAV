"""
clahe_tuning.py — find the best CLAHE parameters for face recognition.

For every (clip_limit, tile_grid_size) combination:
  1. Apply CLAHE + Z-score normalisation to a small dataset subset.
  2. Reduce with PCA (N_PCA_COMPONENTS components).
  3. Classify with 1-NN on a held-out validation split.
  4. Report validation accuracy.

Outputs two figures:
  • Accuracy heatmap  (clip_limit × tile_grid_size)
  • Visual grid       (one sample face under every parameter combo)
"""

import sys
import re
import zipfile
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main_structure"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from face_id.data import _apply_clahe
from face_id.config import DATASET_PATH

# ── experiment settings ────────────────────────────────────────────────────────
CLIP_LIMITS      = [1.0, 2.0, 3.0, 5.0]
TILE_GRID_SIZES  = [(4, 4), (8, 8), (16, 16)]
MAX_PER_PERSON   = 12   # images per person to load
MAX_PEOPLE       = 25   # how many different people to include
N_PCA_COMPONENTS = 50
RANDOM_SEED      = 42
SAMPLE_PERSON_IDX = 0   # index into people list for the visual grid
# ──────────────────────────────────────────────────────────────────────────────

PGM_PATTERN = re.compile(r"(?:.*/)?p(\d+)_i(\d+)(?:\(\d+\))?\.pgm$")


# ─── data helpers ─────────────────────────────────────────────────────────────

def _read_pgm_bytes(f):
    magic = f.readline().strip()
    line = f.readline()
    while line.strip().startswith(b"#"):
        line = f.readline()
    w, h = map(int, line.split())
    maxval = int(f.readline().strip())
    dtype = np.uint8 if maxval < 256 else np.uint16
    return np.frombuffer(f.read(), dtype=dtype).reshape(h, w)


def _crop_gray_bar(img):
    """Crop 64×72 → 64×64 (same logic as data.preprocess_image)."""
    if img.shape == (64, 72):
        if np.var(img[:, :8]) < np.var(img[:, -8:]):
            return img[:, 8:]
        return img[:, :64]
    return img


def load_raw_images(dataset_path, max_per_person, max_people, seed):
    """Load raw uint8 images (gray-bar-cropped only, no CLAHE/normalise)."""
    grouped = {}
    with zipfile.ZipFile(dataset_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".pgm")]
        for m in members:
            match = PGM_PATTERN.match(m)
            if match:
                grouped.setdefault(int(match.group(1)), []).append(m)

        rng = np.random.default_rng(seed)
        people = sorted(grouped)[:max_people]
        raw_images, labels = [], []

        for label_idx, pid in enumerate(people):
            paths = grouped[pid]
            n_pick = min(max_per_person, len(paths))
            chosen = rng.choice(len(paths), size=n_pick, replace=False)
            for i in chosen:
                with zf.open(paths[i]) as f:
                    img = _read_pgm_bytes(f)
                raw_images.append(_crop_gray_bar(img))
                labels.append(label_idx)

    return raw_images, np.array(labels, dtype=np.int32), people


# ─── feature extraction ────────────────────────────────────────────────────────

def make_features(raw_images, clip_limit, tile_grid_size):
    """CLAHE → Z-score → flatten for every image."""
    vecs = []
    for img in raw_images:
        eq = _apply_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        f = eq.astype(np.float32)
        m, s = f.mean(), f.std()
        vecs.append(((f - m) / s if s > 0 else f - m).ravel())
    return np.array(vecs, dtype=np.float32)


# ─── quick PCA + 1-NN evaluator ───────────────────────────────────────────────

def _pca(X_train, n_components):
    mean = X_train.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
    return mean, Vt[:n_components]


def _1nn_accuracy(Z_train, y_train, Z_val, y_val):
    # vectorised squared-distance matrix
    dists = np.sum((Z_val[:, None] - Z_train[None, :]) ** 2, axis=2)
    nn_labels = y_train[np.argmin(dists, axis=1)]
    return float((nn_labels == y_val).mean())


def evaluate(X, y, n_pca, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    X_tr, X_val = X[idx[:split]], X[idx[split:]]
    y_tr, y_val = y[idx[:split]], y[idx[split:]]

    mean, components = _pca(X_tr, n_pca)
    Z_tr  = (X_tr  - mean) @ components.T
    Z_val = (X_val - mean) @ components.T
    return _1nn_accuracy(Z_tr, y_tr, Z_val, y_val)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading raw images (max {MAX_PER_PERSON}/person, {MAX_PEOPLE} people)...")
    raw_images, labels, people = load_raw_images(
        DATASET_PATH, MAX_PER_PERSON, MAX_PEOPLE, RANDOM_SEED
    )
    n_classes = len(np.unique(labels))
    print(f"  Loaded {len(raw_images)} images · {n_classes} people\n")

    combos = list(product(CLIP_LIMITS, TILE_GRID_SIZES))
    results = {}

    for clip_limit, tile_grid in combos:
        tag = f"clip={clip_limit}  tiles={tile_grid[0]}×{tile_grid[1]}"
        print(f"  Testing {tag} ...", end="", flush=True)
        X = make_features(raw_images, clip_limit, tile_grid)
        acc = evaluate(X, labels, N_PCA_COMPONENTS, RANDOM_SEED)
        results[(clip_limit, tile_grid)] = acc
        print(f"  val_acc = {acc:.4f}")

    # ── best result ────────────────────────────────────────────────────────────
    best = max(results, key=results.get)
    print(f"\n{'─'*55}")
    print(f"  BEST: clip_limit={best[0]}, tile_grid_size={best[1]}")
    print(f"        val_acc = {results[best]:.4f}")
    print(f"{'─'*55}\n")

    # ── Figure 1: accuracy heatmap ─────────────────────────────────────────────
    acc_matrix = np.array(
        [[results[(cl, tg)] for tg in TILE_GRID_SIZES] for cl in CLIP_LIMITS]
    )
    tile_labels = [f"{tg[0]}×{tg[1]}" for tg in TILE_GRID_SIZES]
    clip_labels  = [str(cl) for cl in CLIP_LIMITS]

    fig1, ax = plt.subplots(figsize=(7, 5))
    fig1.suptitle(
        f"CLAHE Tuning — Validation Accuracy\n"
        f"(PCA-{N_PCA_COMPONENTS} + 1-NN, {n_classes} people, "
        f"{len(raw_images)} images)",
        fontsize=13, fontweight="bold",
    )
    im = ax.imshow(acc_matrix, cmap="YlGn",
                   vmin=acc_matrix.min() - 0.02,
                   vmax=acc_matrix.max() + 0.02)
    ax.set_xticks(range(len(TILE_GRID_SIZES)))
    ax.set_xticklabels(tile_labels, fontsize=11)
    ax.set_yticks(range(len(CLIP_LIMITS)))
    ax.set_yticklabels(clip_labels, fontsize=11)
    ax.set_xlabel("Tile Grid Size", fontsize=12)
    ax.set_ylabel("Clip Limit", fontsize=12)

    for r, cl in enumerate(CLIP_LIMITS):
        for c, tg in enumerate(TILE_GRID_SIZES):
            is_best = (cl, tg) == best
            val = acc_matrix[r, c]
            ax.text(
                c, r, f"{val:.3f}{'  ★' if is_best else ''}",
                ha="center", va="center",
                fontsize=12 if not is_best else 13,
                fontweight="bold" if is_best else "normal",
                color="black",
            )

    plt.colorbar(im, ax=ax, label="Validation Accuracy")
    fig1.tight_layout()

    # ── Figure 2: visual comparison grid ──────────────────────────────────────
    # pick one image from the sample person
    sample_mask  = np.where(labels == SAMPLE_PERSON_IDX)[0]
    sample_raw   = raw_images[sample_mask[0]]
    person_label = f"person_{people[SAMPLE_PERSON_IDX]}"

    n_rows = len(TILE_GRID_SIZES)
    n_cols = len(CLIP_LIMITS) + 1  # +1 for original

    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(3.5 * n_cols, 3.5 * n_rows))
    fig2.suptitle(
        f"CLAHE Visual Comparison — {person_label}",
        fontsize=14, fontweight="bold",
    )

    for row, tg in enumerate(TILE_GRID_SIZES):
        # original (column 0)
        axes2[row, 0].imshow(sample_raw, cmap="gray")
        axes2[row, 0].set_ylabel(f"tiles {tg[0]}×{tg[1]}", fontsize=10)
        if row == 0:
            axes2[row, 0].set_title("Original", fontsize=10, fontweight="bold")
        axes2[row, 0].axis("off")

        for col, cl in enumerate(CLIP_LIMITS, start=1):
            eq = _apply_clahe(sample_raw, clip_limit=cl, tile_grid_size=tg)
            acc = results[(cl, tg)]
            is_best = (cl, tg) == best
            axes2[row, col].imshow(eq, cmap="gray")
            title = f"clip={cl}" + ("  ★" if is_best else "")
            axes2[row, col].set_title(
                title if row == 0 else ("★ best" if is_best else ""),
                fontsize=10,
                fontweight="bold" if is_best else "normal",
                color="#2a7a2a" if is_best else "black",
            )
            # show accuracy in corner
            axes2[row, col].text(
                2, sample_raw.shape[0] - 3,
                f"acc={acc:.3f}",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
            )
            axes2[row, col].axis("off")

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

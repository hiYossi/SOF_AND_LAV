"""
benchmark_clahe.py
Compares face-recognition accuracy WITH vs WITHOUT CLAHE.
Models: KNN (k=1,3,5) + Linear Least Squares, using PCA reduction.
Target runtime: ~5 minutes.
"""
import sys, re, zipfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main_structure"))

import numpy as np
import matplotlib.pyplot as plt

from face_id.data import _apply_clahe
from face_id.config import DATASET_PATH
from face_id.splits import stratified_k_fold_indices

# ── settings (tune these to hit ~5 min on your machine) ───────────────────────
N_PEOPLE         = 30
MAX_PER_PERSON   = 12    # images per person
N_PCA_COMPONENTS = 50
N_FOLDS          = 3
RANDOM_SEED      = 42
CLAHE_CLIP       = 2.0
CLAHE_TILES      = (8, 8)
KNN_K_VALUES     = [1, 3, 5]
# ──────────────────────────────────────────────────────────────────────────────

PLOT_OUT = Path(__file__).resolve().parent / "benchmark_clahe_result.png"

PGM_PAT = re.compile(r"(?:.*/)?p(\d+)_i(\d+)(?:\(\d+\))?\.pgm$")


def _read_pgm(f):
    f.readline()                          # P5 magic
    line = f.readline()
    while line.strip().startswith(b"#"):
        line = f.readline()
    w, h = map(int, line.split())
    maxval = int(f.readline().strip())
    dtype = np.uint8 if maxval < 256 else np.uint16
    return np.frombuffer(f.read(), dtype=dtype).reshape(h, w)


def _crop(img):
    if img.shape == (64, 72):
        return img[:, 8:] if np.var(img[:, :8]) < np.var(img[:, -8:]) else img[:, :64]
    return img


def load_raw(dataset_path, n_people, max_per, seed):
    grouped = {}
    with zipfile.ZipFile(dataset_path, "r") as zf:
        for m in zf.namelist():
            if m.lower().endswith(".pgm"):
                match = PGM_PAT.match(m)
                if match:
                    grouped.setdefault(int(match.group(1)), []).append(m)

        rng = np.random.default_rng(seed)
        people = sorted(grouped)[:n_people]
        total = sum(min(max_per, len(grouped[p])) for p in people)
        print(f"  Loading {total} images from {len(people)} people...")

        imgs, labels = [], []
        for label_idx, pid in enumerate(people):
            paths = grouped[pid]
            chosen = rng.choice(len(paths), size=min(max_per, len(paths)), replace=False)
            for i in chosen:
                with zf.open(paths[i]) as f:
                    imgs.append(_crop(_read_pgm(f)))
                labels.append(label_idx)
            if (label_idx + 1) % 10 == 0:
                print(f"    {len(imgs)}/{total}", flush=True)

    return imgs, np.array(labels, dtype=np.int32), len(people)


def make_features(raw_imgs, use_clahe):
    vecs = []
    for img in raw_imgs:
        if use_clahe:
            img = _apply_clahe(img, clip_limit=CLAHE_CLIP, tile_grid_size=CLAHE_TILES)
        f = img.astype(np.float32)
        m, s = f.mean(), f.std()
        vecs.append(((f - m) / s if s > 0 else f - m).ravel())
    return np.array(vecs, dtype=np.float32)


def fit_pca(X_train, n):
    mean = X_train.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
    return mean, Vt[:n]


def knn_acc(Z_tr, y_tr, Z_te, y_te, k):
    dists = np.sum((Z_te[:, None] - Z_tr[None, :]) ** 2, axis=2)
    if k == 1:
        preds = y_tr[np.argmin(dists, axis=1)]
    else:
        nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
        n_cls = int(y_tr.max()) + 1
        preds = np.array([
            np.bincount(y_tr[nn_idx[i]], minlength=n_cls).argmax()
            for i in range(len(Z_te))
        ])
    return float((preds == y_te).mean())


def linear_ls_acc(Z_tr, y_tr, Z_te, y_te, n_classes):
    Y = np.zeros((len(y_tr), n_classes), dtype=np.float32)
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    Z1 = np.hstack([Z_tr, np.ones((len(Z_tr), 1), dtype=np.float32)])
    W = np.linalg.pinv(Z1) @ Y
    Z1_te = np.hstack([Z_te, np.ones((len(Z_te), 1), dtype=np.float32)])
    return float((np.argmax(Z1_te @ W, axis=1) == y_te).mean())


def run_cv(X, y, n_classes, label):
    splits = stratified_k_fold_indices(y, N_FOLDS, seed=RANDOM_SEED)
    scores = {f"KNN k={k}": [] for k in KNN_K_VALUES}
    scores["Linear LS"] = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits, 1):
        print(f"  [{label}] Fold {fold_i}/{N_FOLDS} ...", end="", flush=True)
        t0 = time.time()
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        mean, comps = fit_pca(X_tr, N_PCA_COMPONENTS)
        Z_tr = (X_tr - mean) @ comps.T
        Z_te = (X_te - mean) @ comps.T

        for k in KNN_K_VALUES:
            scores[f"KNN k={k}"].append(knn_acc(Z_tr, y_tr, Z_te, y_te, k))
        scores["Linear LS"].append(linear_ls_acc(Z_tr, y_tr, Z_te, y_te, n_classes))
        print(f" {time.time()-t0:.1f}s")

    return {name: (np.mean(v), np.std(v)) for name, v in scores.items()}


def print_table(r_no, r_cl, models):
    print(f"\n{'='*62}")
    print(f"{'Model':<14} {'No CLAHE':>16} {'With CLAHE':>16} {'Diff':>10}")
    print(f"{'-'*62}")
    for name in models:
        nc_m, nc_s = r_no[name]
        cl_m, cl_s = r_cl[name]
        delta = cl_m - nc_m
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:<14} {nc_m*100:>6.2f}% ±{nc_s*100:.2f}%"
            f"  {cl_m*100:>6.2f}% ±{cl_s*100:.2f}%"
            f"  {sign}{delta*100:.2f}%"
        )
    print(f"{'='*62}")


def plot_results(r_no, r_cl, models, n_people, n_images):
    x  = np.arange(len(models))
    w  = 0.35
    means_no = [r_no[m][0] * 100 for m in models]
    stds_no  = [r_no[m][1] * 100 for m in models]
    means_cl = [r_cl[m][0] * 100 for m in models]
    stds_cl  = [r_cl[m][1] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w/2, means_no, w, yerr=stds_no, capsize=5,
                   label="No CLAHE",  color="#5b8db8", alpha=0.9)
    bars2 = ax.bar(x + w/2, means_cl, w, yerr=stds_cl, capsize=5,
                   label=f"With CLAHE (clip={CLAHE_CLIP}, tiles={CLAHE_TILES[0]}x{CLAHE_TILES[1]})",
                   color="#e07b39", alpha=0.9)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Face Recognition: CLAHE vs No CLAHE\n"
        f"PCA-{N_PCA_COMPONENTS}, {N_FOLDS}-fold CV — {n_people} people, {n_images} images",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, min(100, max(max(means_no), max(means_cl)) * 1.18))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {PLOT_OUT}")
    plt.show()


def main():
    t_start = time.time()
    print("=" * 60)
    print(" CLAHE Benchmark")
    print("=" * 60)

    print("\n[1/4] Loading raw images...")
    t0 = time.time()
    raw_imgs, labels, n_people = load_raw(DATASET_PATH, N_PEOPLE, MAX_PER_PERSON, RANDOM_SEED)
    n_classes = len(np.unique(labels))
    print(f"      Done ({time.time()-t0:.1f}s) — {len(raw_imgs)} images, {n_people} people\n")

    print("[2/4] Building features WITHOUT CLAHE...")
    t0 = time.time()
    X_no = make_features(raw_imgs, use_clahe=False)
    print(f"      Done ({time.time()-t0:.1f}s)\n")

    print(f"[3/4] Building features WITH CLAHE (clip={CLAHE_CLIP}, tiles={CLAHE_TILES})...")
    t0 = time.time()
    X_cl = make_features(raw_imgs, use_clahe=True)
    print(f"      Done ({time.time()-t0:.1f}s)\n")

    print(f"[4/4] Running {N_FOLDS}-fold CV on both pipelines (PCA-{N_PCA_COMPONENTS})...\n")
    r_no = run_cv(X_no, labels, n_classes, "No CLAHE")
    print()
    r_cl = run_cv(X_cl, labels, n_classes, "CLAHE   ")

    models = list(r_no.keys())
    print_table(r_no, r_cl, models)
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n", flush=True)

    plot_results(r_no, r_cl, models, n_people, len(raw_imgs))


if __name__ == "__main__":
    main()

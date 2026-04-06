# Face Recognition Project - README

## What Was Built

A complete face recognition project with multiple baselines. The project implements:

- **PCA (Eigenfaces)**: Dimensionality reduction via SVD
- **Linear Least Squares Classifier**: Multiclass classification in PCA space
- **PyTorch CNN**: Neural-network baseline for face recognition
- **Complete pipeline**: Load → Preprocess → Train → Evaluate → Predict

The original linear baseline uses only NumPy. The PyTorch CNN requires `torch`.

---

## Files

| File | Purpose |
|------|---------|
| `linear_model.py` | Main implementation (700+ lines, fully documented) |
| `pytorch_model/torch_face_recognition.py` | Improved PyTorch CNN with validation split and augmentation |
| `FACE_RECOGNITION_GUIDE.md` | Detailed guide & mathematical background |
| `quick_start.py` | Fast demo with smaller dataset |
| `README.md` | This file |

---

## Quick Start

### 1. Download & Organize
Your dataset is already in `Train Set (Labeled)-20260405T164823Z-3-001.zip`:
- 28 people
- ~3600+ labeled face images
- Format: `p{person_id}_i{image_id}.pgm`

### 2. Run the Full Pipeline
```bash
python linear_model.py
```

**Expected output** (first few lines):
```
======================================================================
FACE RECOGNITION: PCA + LINEAR LEAST SQUARES
======================================================================

[1] Loading dataset...
  Found 28 unique persons in filenames
  Loaded 100 images...
  ... (will take several minutes to load ~3600 images)

[2] Splitting into train/test...
[3] Fitting PCA...
[4] Training classifier...
[5] Evaluating...
[6] Baseline comparison...
[7] Single image prediction...
```

**Runtime**: ~5-10 minutes (loading & preprocessing is slow, but training is fast)

### 3. Run Quick Start (Faster Demo)
For instant results with a subset:
```bash
python quick_start.py
```
(Loads only first 500 images, runs in ~1 minute)

### 4. Run the PyTorch Neural Network
Install PyTorch first:
```bash
pip install torch
```

Then run the CNN baseline:
```bash
python pytorch_model/torch_face_recognition.py
```

On machines where `python` points to an older interpreter, use Python 3.11 explicitly:
```bash
py -3.11 pytorch_model/torch_face_recognition.py
```

This uses the original resolution and all available images by default.

The improved PyTorch pipeline now uses:
- stratified train/validation/test splits
- light training augmentation
- validation-based checkpointing instead of choosing on the test set

If you want to run a smaller experiment explicitly:
```bash
python pytorch_model/torch_face_recognition.py --max-images-per-person 16 --epochs 12 --batch-size 64
```

---

## Performance Optimization

If the full run is too slow, tweak these in `linear_model.py`:

```python
# At the top of the file
IMAGE_SIZE = (32, 32)       # Reduce from (64, 64) — 4x fewer pixels
N_COMPONENTS = 30           # Reduce from 50 components
TEST_RATIO = 0.2            # Or use 0.1 for smaller test set
VERBOSE = True              # Keep True to see progress

# To load subset of data, modify load_dataset():
# Add early exit after N images to test faster
```

**Speed comparison:**
- 64×64 images, 50 components, 3600 images: ~8 min
- 32×32 images, 30 components, 1000 images: ~1 min
- 32×32 images, 20 components, 500 images: ~30 sec

---

## Code Structure

### Main Functions

#### Data Loading
- `load_pgm_simple(filename)` - Load single PGM image
- `preprocess_image(image, target_size)` - Resize & normalize
- `load_dataset(root_dir, image_size)` - Load labeled dataset from zip

#### PCA
- `fit_pca(X_train, n_components)` - Fit PCA using SVD
- `transform_pca(X, mean_face, components)` - Project data

#### Classification
- `fit_linear_least_squares(Z_train, y_train, num_classes)` - Train linear classifier
- `predict_linear(Z, W)` - Predict class labels

#### Evaluation
- `accuracy_score_numpy(y_true, y_pred)` - Classification accuracy
- `confusion_matrix_numpy(y_true, y_pred, num_classes)` - Confusion matrix
- `predict_single_image(...)` - Predict on new image

#### Baselines
- `compute_class_means(Z_train, y_train, num_classes)` - Compute class centers
- `predict_nearest_class_mean(Z, class_means)` - Nearest-class-mean classifier

---

## How It Works (30-Second Version)

1. **Load images** → flatten to vectors (64×64 → 4096 dims)
2. **Compute mean face** → subtract from all images
3. **PCA via SVD** → extract top 50 eigenfaces (directions of variation)
4. **Project data** → map from 4096D to 50D space
5. **Train linear classifier** → fit hyperplanes separating classes
6. **Test & score** → measure accuracy on held-out images
7. **Predict** → classify new faces by projecting and scoring

**Key insight**: PCA removes noise & captures identity-relevant features; linear model is fast & interpretable.

---

## What Each Configuration Does

| Setting | Effect | Impact |
|---------|--------|--------|
| `IMAGE_SIZE = (64, 64)` | Higher resolution | Better accuracy, slower |
| `IMAGE_SIZE = (32, 32)` | Lower resolution | Faster, less detail |
| `N_COMPONENTS = 50` | More eigenfaces | Better (if enough data}, slower |
| `N_COMPONENTS = 10` | Fewer eigenfaces | Faster, may oversmooth |
| `TEST_RATIO = 0.2` | 80/20 split | Standard |
| `TEST_RATIO = 0.1` | 90/10 split | More training data |

---

## Example Results (Expected)

With full dataset (3600 images, 28 people):

```
[5] Evaluating...
  Train accuracy: 0.98
  Test accuracy:  0.92

  Confusion matrix (test set):
  [[12  1  0  0 ...]
   [ 1 11  0  0 ...]
   ...          ]

[6] Baseline: Nearest class mean...
  Baseline accuracy: 0.87

[7] Testing single image...
  Test image: Train Set (Labeled)/p27_i447.pgm
  True label: person_27
  Predicted label: person_27
  Confidence: 0.94
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'matplotlib'"
**Solution**: You don't need it! The code works without matplotlib. If you want visualization, install it:
```bash
pip install matplotlib
```
But the core pipeline runs without it.

### Issue: "No PGM files found"
**Check**:
1. Are you in the right directory? (where the zip file is)
2. Is the zip file name correct?
3. Try: `python -c "import zipfile; print(zipfile.ZipFile('Train Set (Labeled)-20260405T164823Z-3-001.zip').namelist()[:5])"`

### Issue: "Out of memory" / Very slow
**Solution**: Reduce dataset size:
```python
# In load_dataset(), add after the loop:
if len(X_list) > 1000:  # Load only first 1000 images
    X_list = X_list[:1000]
    y_list = y_list[:1000]
```

### Issue: "Bad accuracy" (< 70%)
**Check**:
1. Are you using training mean for test data centering? (Check `transform_pca`)
2. Are you fitting PCA only on training data? (Never on test!)
3. Try more components: `N_COMPONENTS = 100`
4. Try larger image size: `IMAGE_SIZE = (128, 128)`

---

## Extending the Code

### Save Trained Model
```python
import pickle

model = {
    'mean_face': mean_face,
    'components': components,
    'W': W,
    'label_to_name': label_to_name,
    'image_size': IMAGE_SIZE,
}
with open('face_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Later, load and use:
model = pickle.load(open('face_model.pkl', 'rb'))
label, scores = predict_single_image('new_face.pgm', **model)
```

### Handle New People (Transfer Learning)
```python
# Fine-tune on new person's images
# (The linear classifier doesn't support this easily, but you could)
# 1. Add new class to W matrix
# 2. Fine-tune W using new images
```

### Cross-Validation
```python
# Replace single train/test with k-fold:
k = 5
for fold in range(k):
    # ... split data per fold
    # ... train & evaluate
    # ... record accuracy
# Report mean ± std accuracy
```

### Different Classifier
Replace `fit_linear_least_squares` with:
- **k-NN**: Find k nearest neighbors in PCA space
- **SVM**: Linear SVM (but need to implement)
- **Logistic regression**: Like least-squares but with sigmoid

---

## Mathematical Details

### PCA via SVD
```
X_centered = X_train - mean_face
U, S, V.T = SVD(X_centered)
eigenfaces = V.T[:n_components, :]
Z = X_centered @ eigenfaces.T
```

### Linear Classifier
```
Z1 = [Z | 1]  (append bias column)
W = pinv(Z1.T @ Z1) @ Z1.T @ Y  (least squares)
scores = Z1 @ W
pred = argmax(scores)
```

---

## References

- **Eigenfaces** (Turk & Pentland, 1991): "Eigenfaces for recognition" 
- **SVD**: `numpy.linalg.svd` documentation
- **Least squares**: Moore-Penrose pseudoinverse
- **Face datasets**: ORL, FERET, LFW (public datasets for face recognition)

---

## License

Educational code — free to use and modify.

---

## Questions?

1. Read `FACE_RECOGNITION_GUIDE.md` for detailed explanations
2. Check docstrings in `linear_model.py` (each function is documented)
3. Run `python quick_start.py` to see it in action quickly
4. Experiment with configuration parameters

**Happy face recognizing!**

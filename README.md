# Exercise 2 - Face Identification

This repo is now organized around the assignment structure from `Exercise 2.pdf`, with only the implemented parts kept:

- `Part A`: data representation and preprocessing
- `Part B`: supervised model comparison
- `Part D`: nested K-fold model selection

The shared code lives in the `face_id/` package, and each implemented part has its own runnable entry point.

## Project Layout

```text
face_id/
  config.py
  data.py
  metrics.py
  model_registry.py
  part_a.py
  part_b.py
  part_d.py
  pca.py
  selection.py
  splits.py
  models/
    linear_least_squares.py
    nearest_class_mean.py
    knn.py

run_part_a.py
run_part_b.py
run_part_d.py
```

## Supported Models

The classical models currently supported are:

- `linear_least_squares`
- `nearest_class_mean`
- `knn`

`Part D` is separate from the model implementations, so you can run nested K-fold CV on any one of these models or on all of them together.

## How to Run

### Part A

```bash
py -3.11 run_part_a.py
```

Useful options:

```bash
py -3.11 run_part_a.py --max-images-per-person 0
py -3.11 run_part_a.py --image-height 64 --image-width 72
```

### Part B

Compare all supervised models on a single stratified validation split:

```bash
py -3.11 run_part_b.py
```

Run Part B on one specific model:

```bash
py -3.11 run_part_b.py --models knn
```

### Part D

Run nested K-fold model selection on all models:

```bash
py -3.11 run_part_d.py
```

Run nested K-fold on one specific model:

```bash
py -3.11 run_part_d.py --models linear_least_squares
py -3.11 run_part_d.py --models nearest_class_mean
py -3.11 run_part_d.py --models knn
```

Run a smaller fast experiment:

```bash
py -3.11 run_part_d.py --max-images-per-person 4 --outer-folds 2 --inner-folds 2 --component-grid 2 3 --knn-neighbors 1 3
```

## Notes

- By default, the scripts use a balanced per-person cap so experiments stay practical.
- Use `--max-images-per-person 0` if you want to use all labeled images.
- Images are kept at their original resolution by default unless you pass `--image-height` and `--image-width`.
- Preprocessing is shared across all parts through `face_id/data.py`.
- Nested CV and hyperparameter tuning are shared through `face_id/selection.py`.

## Quick Verification

The refactor was checked with:

- Python compile of all new modules and runners
- A small Part B smoke test on all models
- A small Part D nested-CV smoke test on each model separately

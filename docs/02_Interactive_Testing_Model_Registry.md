# Interactive Testing And Model Registry

This document tracks the planned implementation for direct hand-drawn digit
testing, safer model storage, model selection, and model comparison.

## Goals

- Draw a digit locally and test it directly against a trained MNIST model.
- Make the drawing workflow visually close to MNIST: bright digit, dark
  background, centered content, and a 28x28 model input preview.
- Store trained models safely without accidentally overwriting earlier runs.
- Select registered models by ID or choose the latest registered model.
- Compare multiple saved models on the MNIST test split and on a local
  hand-drawn dataset when available.

## Slice Checklist

1. Documentation slice
   - Create this document.
   - Link it from the README.
2. Shared preprocessing slice
   - Add reusable preprocessing for image files and drawn image arrays.
   - Convert inputs to MNIST-like `(1, 28, 28)` arrays.
   - Preserve compatibility with `data/examples/*.png`.
3. Model registry slice
   - Store runs in `models/runs/<run_id>/`.
   - Save `model.keras` and `metadata.json`.
   - Include model checksum and training/evaluation metadata.
   - Prevent accidental overwrite of existing runs.
4. CLI integration slice
   - Add registered save/load options.
   - Add model selection by `--model-id latest|<run_id>`.
   - Add model comparison mode.
5. Tkinter drawing tester slice
   - Add a local canvas-based digit tester.
   - Support predict, clear, save sample, and model selection.
6. Comparison dataset slice
   - Use `data/local_digits/<digit>/*.png` for labeled local samples.
   - Keep personal local drawings ignored by Git.
   - Compare MNIST and local-dataset accuracy side by side.

## Planned Structure

- `mnist_dnn/preprocessing.py`: shared MNIST-style image preparation and local sample
  saving.
- `mnist_dnn/model_registry.py`: safe model run storage, metadata, loading, listing,
  and comparison helpers.
- `scripts/train_dense_mnist.py`: existing CLI entry point, extended through
  `mnist_dnn.core`.
- `scripts/draw_digit_tester.py`: local Tkinter drawing and prediction app.
- `models/runs/<run_id>/model.keras`: saved Keras model.
- `models/runs/<run_id>/metadata.json`: metadata for the saved model.
- `data/local_digits/<digit>/*.png`: optional local labeled comparison samples.

## CLI Commands

Train and save a registered model:

```bash
python scripts/train_dense_mnist.py --mode train --epochs 10 --save-registered --model-name baseline_dense
```

Evaluate the latest registered model:

```bash
python scripts/train_dense_mnist.py --mode evaluate --model-id latest
```

Predict a local image with a selected registered model:

```bash
python scripts/train_dense_mnist.py --mode predict --model-id latest --image-path data/examples/Digit1.png
```

Compare all registered models:

```bash
python scripts/train_dense_mnist.py --mode compare --local-dataset data/local_digits
```

Open the drawing tester:

```bash
python scripts/draw_digit_tester.py --model-id latest
```

## Model Metadata

Each registered run stores:

- `run_id`
- `created_at`
- `model_name`
- `architecture`
- `epochs`
- `learning_rate`
- `mnist_loss`
- `mnist_accuracy`
- `local_loss`
- `local_accuracy`
- `model_checksum`

## Local Dataset Format

Local samples are organized by label:

```text
data/local_digits/
  0/
    sample.png
  1/
    sample.png
  ...
  9/
    sample.png
```

Only folders named `0` through `9` are treated as labeled digit classes.

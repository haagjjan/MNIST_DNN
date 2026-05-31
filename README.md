# MNIST_DNN

Archive and prototype repository for a 2023/24 Maturaarbeit about handwritten digit recognition with an artificial neural network trained on MNIST.

The project is intentionally kept as a cleaned-up historical prototype, not a production ML package. The original Maturaarbeit artefacts are preserved, and a lightly modernized runnable script is provided for GitHub visitors.

## Project Goal

The original goal was to build, understand, optimize, and evaluate a neural network that classifies single handwritten digits from 0 to 9.

The unresolved limitation is generalization: the model performs well on MNIST-style images, but local hand-drawn digits can perform much worse when their stroke width, centering, margins, or handwriting style differ from MNIST.

## Repository Structure

- `src/` - cleaned reusable Python functions for the dense MNIST model.
- `scripts/` - command-line entry points.
- `models/` - local output folder for trained models.
- `data/examples/` - small sample digit images for prediction tests.
- `docs/` - project summary and supporting documentation.
- `archive/` - historical tracked PyCharm project files and old experiments.
- `Projektvereinbarung/` - original project agreement and concept material.
- `MA_WG_2024_Haag_Jan_Werk_17032005.py` - original final script kept unchanged at the repository root.
- `MA_WG_2024_Haag_Jan_17032005.pdf` - final written report.
- `MA_WG_2024_Haag_Jan_Arbeitsjournal_17032005.pdf` - work journal.
- `MA_PPP_Jan.pptx` - presentation slides.

## Model

The cleaned script keeps the same dense architecture as the original optimized project:

1. Flatten 28x28 image input
2. Dense layer, 150 neurons, ReLU
3. Dropout, 0.2
4. Dense layer, 150 neurons, ReLU
5. Dropout, 0.2
6. Dense output layer, 10 neurons, Softmax

Training settings:

- Optimizer: Adam
- Learning rate: `0.002`
- Loss: `sparse_categorical_crossentropy`
- Metric: accuracy

The original root script trains for 1 epoch. The written report discusses results after 10 epochs, so the cleaned CLI defaults to 10 epochs.

## Setup

Use Python 3.11 if possible. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activation is usually:

```bash
.venv\Scripts\activate
```

## Usage

Train a model and save it:

```bash
python scripts/train_dense_mnist.py --mode train --epochs 10 --save-model models/mnist_dense.keras
```

Evaluate a saved model on the MNIST test split:

```bash
python scripts/train_dense_mnist.py --mode evaluate --model-path models/mnist_dense.keras
```

Predict one local digit image:

```bash
python scripts/train_dense_mnist.py --mode predict --model-path models/mnist_dense.keras --image-path data/examples/Digit1.png
```

Predict all supported images in a folder:

```bash
python scripts/train_dense_mnist.py --mode predict --model-path models/mnist_dense.keras --image-folder data/examples
```

## Original Script

`MA_WG_2024_Haag_Jan_Werk_17032005.py` is preserved as the original final Maturaarbeit script. It still contains historical comments and the hardcoded local Windows-style image path used during the original work.

For new runs, prefer:

```bash
python scripts/train_dense_mnist.py --mode train
```

## Results From The Original Work

The written report describes MNIST test accuracy of about 97.5% after 10 epochs with the optimized 150/150-neuron architecture. A field test with locally drawn digits reached about 60%, likely because the local images differed from MNIST in format and handwriting style.

## Limitations

- This is an archival/prototype repository.
- The cleaned script is intentionally small and close to the original project.
- The preprocessing for local images is basic and may not solve the MNIST-to-local-image mismatch.
- Generated model files in `models/` are ignored by Git by default.
- The archived PyCharm files are kept for context, not as the recommended way to run the project.

## Documentation

See `docs/01_Summary.md` for a compact project summary with background, implementation notes, known issues, and sensible next steps.

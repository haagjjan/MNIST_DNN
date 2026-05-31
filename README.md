# MNIST_DNN

Archive and prototype code for a 2023/24 Maturaarbeit project about handwritten digit recognition with an artificial neural network trained on the MNIST dataset.

The original project goal was to build, understand, optimize, and evaluate a neural network that classifies single handwritten digits from 0 to 9.

## What Is Here

- `MA_WG_2024_Haag_Jan_Werk_17032005.py` - main final Python script.
- `MA_WG_2024_Haag_Jan_17032005.pdf` - final written report.
- `MA_WG_2024_Haag_Jan_Arbeitsjournal_17032005.pdf` - work journal.
- `MA_PPP_Jan.pptx` - presentation slides.
- `Informations and Sources.docx` - research notes and source notes.
- `Projektvereinbarung/` - original project agreement, concept, and evaluation material.
- `PycharmProjects/pythonProject/` - older project files, experiments, and a saved TensorFlow model.
- `docs/01_Summary.md` - concise technical and project summary.

## Current Model

The main script uses TensorFlow/Keras with this dense architecture:

1. Flatten 28x28 MNIST image input
2. Dense layer, 150 neurons, ReLU
3. Dropout, 0.2
4. Dense layer, 150 neurons, ReLU
5. Dropout, 0.2
6. Dense output layer, 10 neurons, Softmax

It uses Adam, `sparse_categorical_crossentropy`, accuracy, and a manually set learning rate of `0.002`.

## Results From The Original Work

The written report describes MNIST test accuracy of about 97.5% after 10 epochs with the optimized 150/150-neuron architecture. A field test with locally drawn digits reached about 60%, likely because the local digit images differed from MNIST in stroke width, centering, margins, edge softness, and handwriting style.

## Running Notes

This repository is not yet a clean Python package. There is no dependency lock file or requirements file.

Expected dependencies for the main script:

- Python 3.11
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib

The main script currently trains and predicts immediately when executed. Local digit prediction uses a hardcoded Windows-style path:

```text
\\Users\\Jan_MA\\Downloads\\Jan_MA\\Digits_Example\\Digit<number>.png
```

That path must be changed before local prediction will work on another machine.

## Suggested Cleanup

- Add a reproducible environment file.
- Split training, evaluation, saving, loading, and prediction into separate commands.
- Replace hardcoded paths with command-line arguments or a config file.
- Add a consistent preprocessing pipeline for local digit images.
- Move experiments and generated PyCharm/venv material out of the core project path.
- Compare the current dense ANN with a small CNN for image classification.

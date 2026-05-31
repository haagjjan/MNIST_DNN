# MNIST DNN Project Summary

## 1. Project Core

- This repository contains a 2023/24 Maturaarbeit about handwritten digit recognition with an artificial neural network trained on MNIST.
- Original title: "Handwritten Digit Recognition - ein ANN mit der MNIST Database".
- Project type: technical production / prototype.
- Main task: classify single handwritten digits into the classes 0-9.
- Main value of the project: understanding, implementing, optimizing, and evaluating a basic neural network, not building a production-grade recognizer.

## 2. Motivation And Learning Goal

- The project was motivated by curiosity about modern AI systems and the wish to understand how software can learn patterns instead of following only fixed rules.
- Handwritten digit recognition was chosen because it is concrete, visual, testable, and small enough for a first machine-learning project.
- MNIST was a good dataset because it is well-known, easy to access through TensorFlow/Keras, and suitable for comparing training accuracy with real-world testing.
- The core learning goal was to connect theory and implementation: neural network structure, training, loss, optimization, and model evaluation.
- The project was not mainly about achieving state-of-the-art accuracy. It was about building a working model and understanding why it works, where it fails, and how hyperparameters affect performance.

## 3. Preserved Original Work

- The historical final script remains at the repository root: `MA_WG_2024_Haag_Jan_Werk_17032005.py`.
- The original report, work journal, presentation, and project agreement material are preserved.
- Old tracked PyCharm project files and intermediate scripts were moved to `archive/pycharm-projects/`.
- The original script is kept as historical material and should not be treated as the recommended modern entry point.
- The original script still trains and predicts immediately when executed and still contains a hardcoded local Windows-style image path.

## 4. Cleaned Repository Structure

- `src/`: cleaned reusable Python code.
- `scripts/`: command-line entry point.
- `models/`: local output folder for trained models.
- `data/examples/`: small sample digit PNGs for prediction tests.
- `docs/`: project documentation and summary.
- `archive/`: old tracked PyCharm workspaces, experiments, and saved model files.
- `README.md`: GitHub-facing overview, setup instructions, usage examples, and limitations.
- `requirements.txt`: basic dependency list.
- `.gitignore`: ignores local environments, caches, generated models, and editor files.

## 5. Cleaned Runnable Code

- Main reusable module: `src/mnist_dnn.py`.
- CLI entry point: `scripts/train_dense_mnist.py`.
- The cleaned version uses `argparse`.
- Supported modes: `--mode train`, `--mode evaluate`, `--mode predict`.
- Important options: `--epochs`, `--model-path`, `--image-path`, `--image-folder`, `--save-model`.
- Importing the cleaned module does not train a model automatically.

## 6. Model And Training Setup

- Framework: TensorFlow/Keras.
- Language target: Python 3.11 recommended.
- Dataset: MNIST from `tf.keras.datasets.mnist`.
- Input format: 28x28 grayscale digit image.
- Architecture: `Flatten` -> Dense 150 ReLU -> Dropout 0.2 -> Dense 150 ReLU -> Dropout 0.2 -> Dense 10 Softmax.
- Optimizer: Adam.
- Learning rate: 0.002.
- Loss: `sparse_categorical_crossentropy`.
- Metric: accuracy.
- The historical root script trains for 1 epoch.
- The written report discusses 10-epoch results.
- The cleaned CLI defaults to 10 epochs to match the report more closely.

## 7. Results And Interpretation

- Early activation-function choices produced about 67% accuracy.
- Switching hidden layers to ReLU increased accuracy to over 90%.
- A smaller logical architecture with fewer hidden neurons reached about 94.5% after 10 epochs.
- The optimized dense architecture with 150 neurons per hidden layer reached about 97.5% on MNIST test data after 10 epochs.
- Adjusting the learning rate toward 0.002 improved performance by about 2 percentage points.
- A local field test with 20 self-created digits reached only about 60% accuracy.
- The main result is therefore mixed: MNIST performance was strong for a beginner dense ANN, but real local digit images exposed a generalization problem.

## 8. Main Limitation

- The model generalizes weakly from MNIST-style digits to locally drawn digits.
- The most likely reason is domain mismatch rather than a completely broken model.
- Local digits differed from MNIST in stroke width, centering, margins, edge softness, and handwriting style.
- MNIST digits are already standardized in ways that local images are not.
- A model trained only on MNIST may learn MNIST-specific visual patterns instead of robust handwritten-digit recognition.

## 9. Current State

- The repository is now understandable as an archival/prototype project.
- The original Maturaarbeit work remains available.
- A cleaned runnable script exists for train/evaluate/predict workflows.
- Basic sample images are available in `data/examples/`.
- A basic dependency file exists, but the environment is not fully locked.
- There are no automated tests yet.
- Local image preprocessing is still basic and should be considered experimental.

## 10. Sensible Next Steps

- Improve preprocessing for local digit images: grayscale, resizing, inversion, centering, margin normalization, and stroke-thickness handling.
- Create a small repeatable local evaluation dataset instead of relying on ad hoc field-test images.
- Add lightweight tests for preprocessing and prediction shape.
- Pin the environment more exactly once the target Python/TensorFlow version is chosen.
- Optionally compare this dense ANN with a small CNN, because CNNs are better suited to image tasks.

## 11. Key Takeaway

- The project successfully built and documented a working MNIST digit classifier.
- The important unresolved issue is not basic MNIST accuracy, but transfer to locally drawn digits.
- Future work should focus on reproducibility, preprocessing, and evaluation on non-MNIST images while preserving the project as a historical prototype.

# MNIST DNN Project Summary

## Project Identity

This repository contains a 2023/24 Maturaarbeit project about handwritten digit recognition with an artificial neural network trained on the MNIST dataset. The original work is titled "Handwritten Digit Recognition - ein ANN mit der MNIST Database" and was built as a technical production project.

The central objective was to create, understand, and optimize a neural network that reads digital images of handwritten single digits and classifies them into the decimal classes 0-9.

## Original Goals

- Build an artificial neural network for recognizing handwritten digits.
- Learn the theoretical basics of AI, machine learning, deep learning, feedforward, loss functions, optimizers, gradient descent, and backpropagation.
- Choose a suitable programming language and library stack.
- Optimize selected hyperparameters and evaluate the effect on accuracy.
- Test the trained model both on MNIST test data and on locally created handwritten digit images.

The project deliberately used libraries instead of implementing the neural network from scratch. This was considered acceptable for the original scope because the main work combined implementation, explanation, optimization, and evaluation.

## Main Technical Stack

- Python 3.11 was the chosen language.
- PyCharm and virtual environments were used during development.
- Main libraries used in the final script:
  - TensorFlow
  - Keras
  - OpenCV (`cv2`)
  - NumPy
  - Matplotlib
  - `os`

The original work notes that dependency and version conflicts were a recurring issue, especially when examples from tutorials or forum posts used older TensorFlow/Keras APIs.

## Main Implementation

The main final script is:

- `MA_WG_2024_Haag_Jan_Werk_17032005.py`

There are related copies and intermediate versions under `PycharmProjects/pythonProject/`, including:

- `MA_WG_2024_Haag_Jan_Werk_17032005.py`
- `MA_WG_2024_Haag_Jan_Werk_nocomments.py`
- `MA_JanHaag.py`
- `Copy_MA_JanHaag.py`
- `MA_Beschreibung_Quotes.py`
- `VenvTensorflowDownload.py`

The final root script is the most complete commented version. The `*_nocomments.py` version is the same general implementation with comments removed. `Copy_MA_JanHaag.py` appears to preserve a stronger final training configuration with 10 epochs. Some earlier scripts use 120 neurons or higher dropout values and should be treated as experiments.

## Model Architecture

The final model is a Keras `Sequential` neural network:

1. `Flatten(input_shape=(28, 28))`
2. Dense hidden layer with 150 neurons and ReLU activation
3. Dropout with rate 0.2
4. Dense hidden layer with 150 neurons and ReLU activation
5. Dropout with rate 0.2
6. Dense output layer with 10 neurons and Softmax activation

Compilation settings:

- Optimizer: Adam
- Loss: `sparse_categorical_crossentropy`
- Metric: accuracy
- Learning rate: manually set to `0.002`

Training data:

- TensorFlow/Keras MNIST dataset
- `x_train` and `x_test` normalized with `tf.keras.utils.normalize(..., axis=1)`

The root final script currently trains for 1 epoch. The written report describes accuracy after 10 epochs, and `PycharmProjects/pythonProject/Copy_MA_JanHaag.py` uses 10 epochs.

## Program Structure

The final script is organized into six functions:

- `versions()`: prints TensorFlow, Keras, and NumPy versions.
- `create_ann()`: builds the neural network architecture.
- `compile_ann(model)`: compiles the model and sets optimizer, loss, metrics, and learning rate.
- `fit_ann(model)`: trains the model on MNIST.
- `loss_accuracy(model)`: evaluates loss and accuracy on MNIST test data.
- `predict_input(model)`: loops through local PNG files and predicts each digit.

At the end of the script, these functions are executed immediately. This means importing the file as a module would also trigger training and prediction.

## Local Digit Prediction

The script expects local test images at a hardcoded Windows-style path:

`\\Users\\Jan_MA\\Downloads\\Jan_MA\\Digits_Example\\Digit<number>.png`

The prediction loop starts at `Digit0.png` in the root final script. For each existing PNG, it:

1. Reads the image with OpenCV.
2. Selects one color channel.
3. Inverts the pixel values with NumPy.
4. Runs `model.predict(...)`.
5. Prints the raw output vector.
6. Prints the digit with the highest Softmax value.
7. Shows the image with Matplotlib.

Important implementation detail: different script versions use different OpenCV color channels (`[:,:,0]` or `[:,:,1]`). This should be standardized in future cleanup.

## Reported Results

The written report states:

- Early activation choices gave roughly 67% accuracy.
- Switching hidden layers to ReLU increased accuracy to over 90%.
- A smaller "logical" architecture with 20 and 16 hidden neurons reached about 94.5% after 10 epochs.
- A larger trial-and-error architecture with 150 neurons per hidden layer reached about 97.5% after 10 epochs.
- Adjusting the learning rate from 0.01 toward 0.002 improved performance by about 2 percentage points.
- The MNIST test set result was much better than the field test result.
- A field test with 20 self-created digits from 10 people reached only about 60% accuracy.

The project concluded that the low field-test accuracy was likely not primarily caused by overfitting or underfitting, but by mismatch between MNIST images and the locally drawn digits.

## Main Error Sources Identified

The report identifies these likely causes for weak performance on local digits:

- Different stroke thickness: field-test digits were mostly 1 px, MNIST digits were closer to 3 px.
- Different edge appearance: MNIST digits had softer gray edges, field-test digits had harder black edges.
- Different centering and margins: MNIST digits are centered with a small border, while local digits had inconsistent spacing.
- Different handwriting styles: MNIST is based on US handwriting samples, while local test digits may use regional writing conventions.
- Potential domain mismatch: a model trained only on MNIST generalizes poorly to differently formatted local images.

Suggested remedies from the report:

- Reformat or preprocess local inputs to better match MNIST.
- Expand or adapt the dataset with more local writing examples.
- Use separate models or preprocessing strategies for different input formats or writing styles.

## Important Repository Contents

- `MA_WG_2024_Haag_Jan_Werk_17032005.py`: main final Python script.
- `MA_WG_2024_Haag_Jan_17032005.pdf`: final written report.
- `MA_WG_2024_Haag_Jan_Arbeitsjournal_17032005.pdf`: work journal.
- `MA_PPP_Jan.pptx`: presentation slides.
- `Informations and Sources.docx`: early research notes and source notes.
- `Introduction to neural networks`: JPEG image/reference asset.
- `Projektvereinbarung/`: project agreement, concept, and evaluation material.
- `PycharmProjects/pythonProject/HandwrittenDigitRecognition/`: saved TensorFlow model generated during development.
- `PycharmProjects/000_VenvSetup/` and `PycharmProjects/HelloWorldProject/`: setup and learning experiments, not core project code.

The repository contains PyCharm metadata, generated files, and old virtual environment material. The current `.gitignore` excludes some of these categories, but many files are already tracked.

## Current State

The project is best understood as an archival research and prototype repository, not as a clean reusable Python package yet.

Strengths:

- The final script is readable and heavily commented.
- The model architecture and hyperparameters are clearly visible.
- The written documentation explains the motivation, theory, implementation choices, optimization process, and evaluation.
- A saved model exists in the PyCharm project folder.

Limitations:

- No `requirements.txt`, `pyproject.toml`, or reproducible environment definition.
- Hardcoded local Windows paths for user-created digit images.
- Training and prediction run automatically at script execution time.
- The script is not structured as a reusable module or command-line tool.
- No tests.
- No standardized local image preprocessing pipeline.
- Several duplicated or experimental scripts make it unclear which file is canonical without reading the documentation.
- Some old scripts contain known typos or obsolete TensorFlow API usage.

## Best Next Technical Directions

1. Create a clean project structure with separate folders for source code, data samples, models, notebooks/scripts, and docs.
2. Add a reproducible dependency file.
3. Refactor the final script so training, evaluation, saving, loading, and prediction are separate commands.
4. Replace hardcoded paths with configurable arguments.
5. Add a preprocessing pipeline for local digit images:
   - grayscale conversion
   - resize to 28x28
   - inversion if needed
   - centering
   - border/margin normalization
   - stroke thickness or threshold normalization
6. Save trained models consistently and load them for inference instead of retraining every run.
7. Create a small local evaluation dataset and record predictions in a repeatable way.
8. Compare a dense neural network against a small CNN, because image classification usually benefits from convolutional layers.
9. Add lightweight tests for preprocessing and prediction-shape behavior.
10. Remove or archive duplicate PyCharm/venv material once the clean structure exists.

## Key Takeaway

The original project successfully demonstrated a working MNIST digit classifier and documented a meaningful optimization process. The main unsolved problem is not basic MNIST performance, but robust generalization from MNIST-style images to locally drawn digits. Future work should focus on reproducibility, code cleanup, standardized preprocessing, and evaluation on non-MNIST inputs.

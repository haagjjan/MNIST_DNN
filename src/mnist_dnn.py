"""Clean runnable version of the original MNIST dense ANN prototype.

This module keeps the original project architecture but separates training,
evaluation, saving, loading, preprocessing, and prediction into reusable
functions. Importing it does not train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (28, 28)
DEFAULT_MODEL_PATH = Path("models/mnist_dense.keras")


def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load and normalize the MNIST train/test split."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1).astype("float32")
    x_test = tf.keras.utils.normalize(x_test, axis=1).astype("float32")
    return (x_train, y_train), (x_test, y_test)


def build_model(learning_rate: float = 0.002) -> tf.keras.Model:
    """Build the dense MNIST model used in the original project."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGE_SIZE),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(150, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(150, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="mnist_dense_ann",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model: tf.keras.Model, epochs: int = 10) -> tf.keras.callbacks.History:
    """Train the model on the MNIST training split."""
    (x_train, y_train), _ = load_data()
    return model.fit(x_train, y_train, epochs=epochs)


def evaluate_model(model: tf.keras.Model) -> tuple[float, float]:
    """Evaluate the model on the MNIST test split."""
    _, (x_test, y_test) = load_data()
    loss, accuracy = model.evaluate(x_test, y_test)
    return float(loss), float(accuracy)


def save_model(model: tf.keras.Model, model_path: str | Path) -> None:
    """Save a trained model in Keras format."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Load a previously saved Keras model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Read a digit image and convert it to model input shape `(1, 28, 28)`.

    MNIST uses bright digits on a dark background. If the input image appears to
    use a light background, the image is inverted before normalization.
    """
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file could not be read: {path}")

    if image.mean() > 127:
        image = 255 - image

    if image.shape != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    image = image.astype("float32")
    image = tf.keras.utils.normalize(image, axis=1)
    return np.expand_dims(image, axis=0)


def predict_image(model: tf.keras.Model, image_path: str | Path) -> tuple[int, float, np.ndarray]:
    """Predict one local digit image."""
    image = preprocess_image(image_path)
    probabilities = model.predict(image, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return digit, confidence, probabilities


def iter_images(image_folder: str | Path) -> Iterable[Path]:
    """Yield common image files from a folder in name order."""
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    for pattern in extensions:
        yield from sorted(folder.glob(pattern))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or run prediction with the MNIST dense ANN prototype.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "predict"),
        required=True,
        help="Action to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for --mode train.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a saved Keras model for evaluate/predict.",
    )
    parser.add_argument(
        "--image-path",
        help="Path to one local digit image for --mode predict.",
    )
    parser.add_argument(
        "--image-folder",
        help="Folder of local digit images for --mode predict.",
    )
    parser.add_argument(
        "--save-model",
        help="Where to save the trained model for --mode train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        model = build_model()
        train_model(model, epochs=args.epochs)
        loss, accuracy = evaluate_model(model)
        print(f"MNIST test loss: {loss:.4f}")
        print(f"MNIST test accuracy: {accuracy:.4f}")

        if args.save_model:
            save_model(model, args.save_model)
            print(f"Saved model to: {args.save_model}")
        return

    model = load_model(args.model_path)

    if args.mode == "evaluate":
        loss, accuracy = evaluate_model(model)
        print(f"MNIST test loss: {loss:.4f}")
        print(f"MNIST test accuracy: {accuracy:.4f}")
        return

    if not args.image_path and not args.image_folder:
        raise SystemExit("--mode predict requires --image-path or --image-folder")

    image_paths: list[Path] = []
    if args.image_path:
        image_paths.append(Path(args.image_path))
    if args.image_folder:
        image_paths.extend(iter_images(args.image_folder))

    for image_path in image_paths:
        digit, confidence, _ = predict_image(model, image_path)
        print(f"{image_path}: predicted {digit} ({confidence:.2%})")


if __name__ == "__main__":
    main()

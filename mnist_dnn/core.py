"""Clean runnable version of the original MNIST dense ANN prototype.

This module keeps the original project architecture but separates training,
evaluation, saving, loading, preprocessing, and prediction into reusable
functions. Importing it does not train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from mnist_dnn.model_registry import (
    compare_models,
    get_model_info,
    load_registered_model,
    save_registered_model,
)
from mnist_dnn.preprocessing import (
    IMAGE_SIZE,
    LOCAL_DIGITS_DIR,
    load_local_digit_dataset,
    normalize_for_model,
    preprocess_image_file,
)


DEFAULT_MODEL_PATH = Path("models/mnist_dense.keras")


def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load and normalize the MNIST train/test split."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = normalize_for_model(x_train).astype("float32")
    x_test = normalize_for_model(x_test).astype("float32")
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


def evaluate_local_dataset(
    model: tf.keras.Model,
    dataset_path: str | Path,
) -> tuple[float | None, float | None, int]:
    """Evaluate the model on labeled local samples, if any exist."""
    x_local, y_local = load_local_digit_dataset(dataset_path)
    if len(y_local) == 0:
        return None, None, 0
    loss, accuracy = model.evaluate(x_local, y_local, verbose=0)
    return float(loss), float(accuracy), int(len(y_local))


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
    """Read a digit image and convert it to model input shape `(1, 28, 28)`."""
    return preprocess_image_file(image_path)


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
        choices=("train", "evaluate", "predict", "compare"),
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
        "--learning-rate",
        type=float,
        default=0.002,
        help="Learning rate for --mode train.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a saved Keras model for evaluate/predict.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        help="Registered model ID, or latest. May be repeated for --mode compare.",
    )
    parser.add_argument(
        "--model-name",
        default="mnist_dense_ann",
        help="Human-readable model name stored in registry metadata.",
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
    parser.add_argument(
        "--save-registered",
        action="store_true",
        help="Save the trained model to the safe model registry.",
    )
    parser.add_argument(
        "--local-dataset",
        default=str(LOCAL_DIGITS_DIR),
        help="Local labeled digit dataset for registered training metadata and compare mode.",
    )
    return parser.parse_args()


def _load_selected_model(args: argparse.Namespace) -> tf.keras.Model:
    if args.model_id:
        if len(args.model_id) != 1:
            raise SystemExit("--mode evaluate and --mode predict accept only one --model-id")
        model_id = args.model_id[0]
        try:
            model_info = get_model_info(model_id)
            model = load_registered_model(model_id)
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        print(f"Loaded registered model: {model_info.run_id}")
        return model

    return load_model(args.model_path)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _print_comparison(results: list[dict[str, object]]) -> None:
    header = (
        f"{'run_id':<34} {'model':<18} {'mnist_acc':>10} "
        f"{'mnist_loss':>10} {'local_acc':>10} {'local_n':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{str(row['run_id']):<34} "
            f"{str(row['model_name'])[:18]:<18} "
            f"{_format_metric(row['mnist_accuracy']) :>10} "
            f"{_format_metric(row['mnist_loss']) :>10} "
            f"{_format_metric(row['local_accuracy']) :>10} "
            f"{str(row['local_sample_count']):>7}"
        )


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        model = build_model(learning_rate=args.learning_rate)
        train_model(model, epochs=args.epochs)
        loss, accuracy = evaluate_model(model)
        print(f"MNIST test loss: {loss:.4f}")
        print(f"MNIST test accuracy: {accuracy:.4f}")

        local_loss, local_accuracy, local_count = evaluate_local_dataset(model, args.local_dataset)
        if local_count:
            print(f"Local test loss: {local_loss:.4f}")
            print(f"Local test accuracy: {local_accuracy:.4f} ({local_count} samples)")

        if args.save_model:
            save_model(model, args.save_model)
            print(f"Saved model to: {args.save_model}")

        if args.save_registered:
            run_id = save_registered_model(
                model,
                {
                    "model_name": args.model_name,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "mnist_loss": loss,
                    "mnist_accuracy": accuracy,
                    "local_loss": local_loss,
                    "local_accuracy": local_accuracy,
                    "local_sample_count": local_count,
                },
            )
            print(f"Registered model as: {run_id}")
        return

    if args.mode == "compare":
        try:
            results = compare_models(
                model_ids=args.model_id,
                include_mnist=True,
                local_dataset=args.local_dataset,
            )
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        _print_comparison(results)
        return

    model = _load_selected_model(args)

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

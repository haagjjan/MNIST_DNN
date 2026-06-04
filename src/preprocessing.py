"""Shared preprocessing helpers for MNIST-style digit inputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SIZE = (28, 28)
DIGIT_BOX_SIZE = 20
LOCAL_DIGITS_DIR = Path("data/local_digits")
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _as_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image-like array to an 8-bit grayscale image."""
    array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    elif array.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {array.shape}")

    if array.dtype != np.uint8:
        array = array.astype("float32")
        if array.size and array.max() <= 1.0:
            array *= 255.0
        array = np.clip(array, 0, 255)

    return array.astype("uint8")


def _has_light_background(image: np.ndarray) -> bool:
    """Estimate whether the digit image needs inversion."""
    top = image[0, :]
    bottom = image[-1, :]
    left = image[:, 0]
    right = image[:, -1]
    border = np.concatenate((top, bottom, left, right))
    return float(np.median(border)) > 127.0


def _digit_mask(image: np.ndarray) -> np.ndarray:
    """Create a foreground mask for the bright digit on a dark background."""
    if image.max() == 0:
        return np.zeros_like(image, dtype=bool)

    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = thresholded > 0
    if not mask.any():
        mask = image > max(10, int(image.max()) * 0.1)
    return mask


def _center_digit(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop the foreground and place it into a centered MNIST-sized canvas."""
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return np.zeros(IMAGE_SIZE, dtype="uint8")

    cropped = image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    height, width = cropped.shape
    scale = DIGIT_BOX_SIZE / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.zeros(IMAGE_SIZE, dtype="uint8")
    top = (IMAGE_SIZE[0] - resized_height) // 2
    left = (IMAGE_SIZE[1] - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    canvas[canvas < 8] = 0
    return canvas


def prepare_mnist_style_image(image: np.ndarray) -> np.ndarray:
    """Convert a raw image to a centered 28x28 MNIST-style grayscale image."""
    grayscale = _as_uint8_grayscale(image)
    if _has_light_background(grayscale):
        grayscale = 255 - grayscale

    mask = _digit_mask(grayscale)
    return _center_digit(grayscale, mask)


def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """Normalize an image using the same row-wise L2 style as the training data."""
    array = image.astype("float32")
    norm = np.linalg.norm(array, ord=2, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return array / norm


def preprocess_digit_array(image: np.ndarray) -> np.ndarray:
    """Convert a raw digit image array to model input shape `(1, 28, 28)`."""
    prepared = prepare_mnist_style_image(image)
    normalized = normalize_for_model(prepared)
    return np.expand_dims(normalized, axis=0)


def preprocess_image_file(image_path: str | Path) -> np.ndarray:
    """Read and preprocess a digit image file for model prediction."""
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file could not be read: {path}")
    return preprocess_digit_array(image)


def save_local_digit_sample(
    image: np.ndarray,
    label: int | str,
    dataset_dir: str | Path = LOCAL_DIGITS_DIR,
) -> Path:
    """Save a drawn digit as a labeled local comparison sample."""
    label_int = int(label)
    if label_int < 0 or label_int > 9:
        raise ValueError("Digit label must be between 0 and 9")

    prepared = prepare_mnist_style_image(image)
    output_dir = Path(dataset_dir) / str(label_int)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"digit_{timestamp}.png"
    if not cv2.imwrite(str(output_path), prepared):
        raise OSError(f"Could not write local digit sample: {output_path}")
    return output_path


def iter_labeled_digit_images(dataset_dir: str | Path) -> Iterable[tuple[Path, int]]:
    """Yield `(image_path, label)` pairs from `data/local_digits/<digit>/`."""
    root = Path(dataset_dir)
    if not root.exists():
        return

    for label in range(10):
        label_dir = root / str(label)
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield path, label


def load_local_digit_dataset(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a labeled local digit dataset into model-ready arrays."""
    images: list[np.ndarray] = []
    labels: list[int] = []

    for image_path, label in iter_labeled_digit_images(dataset_dir):
        images.append(preprocess_image_file(image_path)[0])
        labels.append(label)

    if not images:
        return (
            np.empty((0, *IMAGE_SIZE), dtype="float32"),
            np.empty((0,), dtype="int64"),
        )

    return np.stack(images).astype("float32"), np.array(labels, dtype="int64")

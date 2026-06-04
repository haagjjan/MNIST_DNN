"""Safe storage, selection, and comparison helpers for trained models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from mnist_dnn.preprocessing import load_local_digit_dataset, normalize_for_model


REGISTRY_ROOT = Path("models/runs")
MODEL_FILENAME = "model.keras"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class ModelInfo:
    """Metadata summary for a registered model run."""

    run_id: str
    model_name: str
    created_at: str
    model_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower())
    return slug.strip("-") or "model"


def _make_run_id(model_name: str, registry_root: Path) -> str:
    base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify(model_name)}"
    candidate = base
    suffix = 2
    while (registry_root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _architecture_summary(model: Any) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for layer in getattr(model, "layers", []):
        config = layer.get_config() if hasattr(layer, "get_config") else {}
        layers.append(
            {
                "name": getattr(layer, "name", layer.__class__.__name__),
                "type": layer.__class__.__name__,
                "units": config.get("units"),
                "activation": config.get("activation"),
                "rate": config.get("rate"),
            }
        )

    return {
        "model_name": getattr(model, "name", "unknown_model"),
        "layers": layers,
    }


def model_checksum(model_path: str | Path) -> str:
    """Return a SHA-256 checksum for a saved model file."""
    path = Path(model_path)
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_registered_model(
    model: Any,
    metadata: dict[str, Any] | None = None,
    registry_root: str | Path = REGISTRY_ROOT,
) -> str:
    """Save a model as a unique registered run and return its run ID."""
    root = Path(registry_root)
    root.mkdir(parents=True, exist_ok=True)

    metadata_input = dict(metadata or {})
    model_name = str(metadata_input.get("model_name") or getattr(model, "name", "mnist_model"))
    run_id = str(metadata_input.get("run_id") or _make_run_id(model_name, root))
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    model_path = run_dir / MODEL_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    model.save(model_path)

    metadata_output: dict[str, Any] = {
        "run_id": run_id,
        "created_at": metadata_input.get("created_at") or _utc_now(),
        "model_name": model_name,
        "architecture": metadata_input.get("architecture") or _architecture_summary(model),
        "epochs": metadata_input.get("epochs"),
        "learning_rate": metadata_input.get("learning_rate"),
        "mnist_loss": metadata_input.get("mnist_loss"),
        "mnist_accuracy": metadata_input.get("mnist_accuracy"),
        "local_loss": metadata_input.get("local_loss"),
        "local_accuracy": metadata_input.get("local_accuracy"),
        "local_sample_count": metadata_input.get("local_sample_count"),
        "model_checksum": model_checksum(model_path),
    }

    for key, value in metadata_input.items():
        metadata_output.setdefault(key, value)

    metadata_path.write_text(json.dumps(metadata_output, indent=2, sort_keys=True), encoding="utf-8")
    return run_id


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def list_models(registry_root: str | Path = REGISTRY_ROOT) -> list[ModelInfo]:
    """List registered model runs sorted by creation time."""
    root = Path(registry_root)
    if not root.exists():
        return []

    models: list[ModelInfo] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = run_dir / METADATA_FILENAME
        model_path = run_dir / MODEL_FILENAME
        if not metadata_path.exists() or not model_path.exists():
            continue
        metadata = _read_metadata(metadata_path)
        models.append(
            ModelInfo(
                run_id=str(metadata.get("run_id", run_dir.name)),
                model_name=str(metadata.get("model_name", run_dir.name)),
                created_at=str(metadata.get("created_at", "")),
                model_path=model_path,
                metadata_path=metadata_path,
                metadata=metadata,
            )
        )

    return sorted(models, key=lambda info: (info.created_at, info.run_id))


def get_model_info(model_id: str, registry_root: str | Path = REGISTRY_ROOT) -> ModelInfo:
    """Resolve a concrete model ID or `latest` to registry metadata."""
    models = list_models(registry_root)
    if not models:
        raise FileNotFoundError(f"No registered models found in {Path(registry_root)}")

    if model_id == "latest":
        return models[-1]

    for model_info in models:
        if model_info.run_id == model_id:
            return model_info

    raise FileNotFoundError(f"Registered model not found: {model_id}")


def load_registered_model(model_id: str, registry_root: str | Path = REGISTRY_ROOT):
    """Load a registered Keras model by run ID or `latest`."""
    import tensorflow as tf

    model_info = get_model_info(model_id, registry_root)
    return tf.keras.models.load_model(model_info.model_path)


def _evaluate_mnist(model: Any) -> tuple[float, float, int]:
    import tensorflow as tf

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = normalize_for_model(x_test).astype("float32")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return float(loss), float(accuracy), int(len(y_test))


def _evaluate_arrays(model: Any, x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float, int]:
    loss, accuracy = model.evaluate(x_values, y_values, verbose=0)
    return float(loss), float(accuracy), int(len(y_values))


def compare_models(
    model_ids: Sequence[str] | None = None,
    include_mnist: bool = True,
    local_dataset: str | Path | None = None,
    registry_root: str | Path = REGISTRY_ROOT,
) -> list[dict[str, Any]]:
    """Compare registered models on MNIST and optionally a local digit dataset."""
    selected_ids = list(model_ids or [])
    if selected_ids:
        model_infos = [get_model_info(model_id, registry_root) for model_id in selected_ids]
    else:
        model_infos = list_models(registry_root)

    if not model_infos:
        raise FileNotFoundError(f"No registered models found in {Path(registry_root)}")

    local_data: tuple[np.ndarray, np.ndarray] | None = None
    if local_dataset:
        x_local, y_local = load_local_digit_dataset(local_dataset)
        if len(y_local) > 0:
            local_data = (x_local, y_local)

    results: list[dict[str, Any]] = []
    for model_info in model_infos:
        model = load_registered_model(model_info.run_id, registry_root)
        row: dict[str, Any] = {
            "run_id": model_info.run_id,
            "model_name": model_info.model_name,
            "created_at": model_info.created_at,
            "mnist_loss": None,
            "mnist_accuracy": None,
            "mnist_sample_count": None,
            "local_loss": None,
            "local_accuracy": None,
            "local_sample_count": 0,
        }

        if include_mnist:
            loss, accuracy, count = _evaluate_mnist(model)
            row["mnist_loss"] = loss
            row["mnist_accuracy"] = accuracy
            row["mnist_sample_count"] = count

        if local_data is not None:
            local_loss, local_accuracy, local_count = _evaluate_arrays(model, *local_data)
            row["local_loss"] = local_loss
            row["local_accuracy"] = local_accuracy
            row["local_sample_count"] = local_count

        results.append(row)

    return results

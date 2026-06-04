from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mnist_dnn.model_registry import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    get_model_info,
    list_models,
    save_registered_model,
)


class FakeLayer:
    name = "dense"

    def get_config(self) -> dict[str, object]:
        return {"units": 10, "activation": "softmax"}


class FakeModel:
    name = "fake_model"
    layers = [FakeLayer()]

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"fake keras model")


class ModelRegistryTests(unittest.TestCase):
    def test_save_registered_model_writes_model_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = save_registered_model(
                FakeModel(),
                {
                    "model_name": "Test Model",
                    "epochs": 3,
                    "mnist_accuracy": 0.975,
                    "created_at": "2026-06-04T10:00:00Z",
                },
                registry_root=temp_dir,
            )

            run_dir = Path(temp_dir) / run_id
            metadata = json.loads((run_dir / METADATA_FILENAME).read_text(encoding="utf-8"))

            self.assertTrue((run_dir / MODEL_FILENAME).exists())
            self.assertEqual(metadata["run_id"], run_id)
            self.assertEqual(metadata["model_name"], "Test Model")
            self.assertEqual(metadata["epochs"], 3)
            self.assertEqual(len(metadata["model_checksum"]), 64)

    def test_explicit_run_id_cannot_overwrite_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = {"run_id": "fixed-run", "model_name": "Fixed"}
            save_registered_model(FakeModel(), metadata, registry_root=temp_dir)

            with self.assertRaises(FileExistsError):
                save_registered_model(FakeModel(), metadata, registry_root=temp_dir)

    def test_list_models_and_latest_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            save_registered_model(
                FakeModel(),
                {
                    "run_id": "older",
                    "model_name": "Older",
                    "created_at": "2026-06-04T10:00:00Z",
                },
                registry_root=temp_dir,
            )
            save_registered_model(
                FakeModel(),
                {
                    "run_id": "newer",
                    "model_name": "Newer",
                    "created_at": "2026-06-04T11:00:00Z",
                },
                registry_root=temp_dir,
            )

            models = list_models(temp_dir)
            latest = get_model_info("latest", temp_dir)

        self.assertEqual([model.run_id for model in models], ["older", "newer"])
        self.assertEqual(latest.run_id, "newer")


if __name__ == "__main__":
    unittest.main()

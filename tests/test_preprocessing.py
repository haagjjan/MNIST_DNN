from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.preprocessing import (
    load_local_digit_dataset,
    prepare_mnist_style_image,
    preprocess_digit_array,
    save_local_digit_sample,
)


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_digit_array_returns_model_shape_and_range(self) -> None:
        image = np.zeros((120, 120), dtype="uint8")
        cv2.line(image, (60, 15), (60, 105), color=255, thickness=18, lineType=cv2.LINE_AA)

        processed = preprocess_digit_array(image)

        self.assertEqual(processed.shape, (1, 28, 28))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreater(float(processed.max()), 0.0)
        self.assertLessEqual(float(processed.max()), 1.0)

    def test_light_background_image_is_inverted_and_centered(self) -> None:
        image = np.full((100, 100), 255, dtype="uint8")
        cv2.circle(image, (50, 50), 28, color=0, thickness=-1, lineType=cv2.LINE_AA)

        prepared = prepare_mnist_style_image(image)

        self.assertEqual(prepared.shape, (28, 28))
        self.assertGreater(int(prepared.max()), 0)
        self.assertEqual(int(prepared[0, 0]), 0)
        self.assertGreater(int(prepared[14, 14]), 0)

    def test_save_and_load_local_digit_dataset(self) -> None:
        image = np.zeros((80, 80), dtype="uint8")
        cv2.line(image, (20, 10), (60, 70), color=255, thickness=14, lineType=cv2.LINE_AA)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = save_local_digit_sample(image, 7, temp_dir)
            self.assertTrue(Path(output_path).exists())

            x_values, y_values = load_local_digit_dataset(temp_dir)

        self.assertEqual(x_values.shape, (1, 28, 28))
        self.assertEqual(y_values.tolist(), [7])


if __name__ == "__main__":
    unittest.main()

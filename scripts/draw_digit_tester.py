"""Local Tkinter app for drawing and testing MNIST-style digits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


CANVAS_SIZE = 280
PREVIEW_SCALE = 4
BRUSH_WIDTH = 22
DEFAULT_MODEL_PATH = Path("models/mnist_dense.keras")
LOCAL_DIGITS_DIR = Path("data/local_digits")


class DigitTesterApp:
    """Canvas-based digit tester for registered or path-based models."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.dataset_dir = Path(args.local_dataset)
        self.drawing = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype="uint8")
        self.last_point: tuple[int, int] | None = None
        self.model = None
        self.loaded_model_key: tuple[str, str] | None = None
        self.model_options: dict[str, tuple[str, str]] = {}

        self.root.title("MNIST Digit Tester")
        self.root.resizable(False, False)
        self._build_ui()
        self._load_model_options()
        self._select_default_model()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(
            main,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.canvas.grid(row=0, column=0, rowspan=8, padx=(0, 12), sticky="n")
        self.canvas.bind("<ButtonPress-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._end_draw)

        ttk.Label(main, text="Model").grid(row=0, column=1, sticky="w")
        self.model_selector = ttk.Combobox(main, width=34, state="readonly")
        self.model_selector.grid(row=1, column=1, sticky="ew", pady=(2, 8))
        self.model_selector.bind("<<ComboboxSelected>>", self._model_changed)

        button_row = ttk.Frame(main)
        button_row.grid(row=2, column=1, sticky="ew", pady=(0, 10))
        ttk.Button(button_row, text="Predict", command=self.predict).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(button_row, text="Clear", command=self.clear).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(button_row, text="Save Sample", command=self.save_sample).grid(row=0, column=2)

        ttk.Label(main, text="28x28 input").grid(row=3, column=1, sticky="w")
        preview_size = 28 * PREVIEW_SCALE
        self.preview = tk.Canvas(
            main,
            width=preview_size,
            height=preview_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.preview.grid(row=4, column=1, sticky="w", pady=(2, 10))

        self.result_var = tk.StringVar(value="Draw a digit, then predict.")
        ttk.Label(main, textvariable=self.result_var, font=("", 14, "bold")).grid(
            row=5,
            column=1,
            sticky="w",
            pady=(0, 8),
        )

        self.probability_vars: list[tk.StringVar] = []
        self.probability_bars: list[ttk.Progressbar] = []
        probabilities = ttk.Frame(main)
        probabilities.grid(row=6, column=1, sticky="ew")
        for digit in range(10):
            label = ttk.Label(probabilities, text=str(digit), width=2)
            label.grid(row=digit, column=0, sticky="w")
            bar = ttk.Progressbar(probabilities, maximum=1.0, length=180)
            bar.grid(row=digit, column=1, padx=6, pady=1)
            value_var = tk.StringVar(value="0.0%")
            ttk.Label(probabilities, textvariable=value_var, width=7).grid(row=digit, column=2, sticky="e")
            self.probability_bars.append(bar)
            self.probability_vars.append(value_var)

        self.status_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.status_var).grid(row=7, column=1, sticky="w", pady=(8, 0))

    def _load_model_options(self) -> None:
        self.model_options.clear()
        registered_models = list_models()
        if registered_models:
            latest = registered_models[-1]
            self.model_options[f"latest ({latest.run_id})"] = ("registry", "latest")
            for model_info in reversed(registered_models):
                label = f"{model_info.run_id} ({model_info.model_name})"
                self.model_options[label] = ("registry", model_info.run_id)

        model_path = Path(self.args.model_path)
        if model_path.exists():
            self.model_options[f"file: {model_path}"] = ("file", str(model_path))

        labels = list(self.model_options)
        self.model_selector["values"] = labels
        if not labels:
            self.status_var.set("No registered model or model file found.")

    def _select_default_model(self) -> None:
        labels = list(self.model_options)
        if not labels:
            return

        requested_model_id = self.args.model_id
        for label, option in self.model_options.items():
            source, value = option
            if source == "registry" and value == requested_model_id:
                self.model_selector.set(label)
                return

        for label, option in self.model_options.items():
            source, value = option
            if source == "registry" and requested_model_id == "latest" and value == "latest":
                self.model_selector.set(label)
                return

        self.model_selector.set(labels[0])

    def _model_changed(self, _event: tk.Event) -> None:
        self.model = None
        self.loaded_model_key = None
        self.status_var.set("Model selection changed.")

    def _start_draw(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)
        self._draw_dot(event.x, event.y)

    def _draw(self, event: tk.Event) -> None:
        if self.last_point is None:
            self.last_point = (event.x, event.y)
        x0, y0 = self.last_point
        x1, y1 = event.x, event.y
        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill="white",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
            tags="stroke",
        )
        cv2.line(
            self.drawing,
            (x0, y0),
            (x1, y1),
            color=255,
            thickness=BRUSH_WIDTH,
            lineType=cv2.LINE_AA,
        )
        self.last_point = (x1, y1)
        self._draw_preview()

    def _draw_dot(self, x: int, y: int) -> None:
        radius = BRUSH_WIDTH // 2
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            outline="white",
            tags="stroke",
        )
        cv2.circle(self.drawing, (x, y), radius, color=255, thickness=-1, lineType=cv2.LINE_AA)
        self._draw_preview()

    def _end_draw(self, _event: tk.Event) -> None:
        self.last_point = None

    def _draw_preview(self) -> None:
        prepared = prepare_mnist_style_image(self.drawing)
        self.preview.delete("all")
        for row in range(28):
            for col in range(28):
                value = int(prepared[row, col])
                if value == 0:
                    continue
                color = f"#{value:02x}{value:02x}{value:02x}"
                x0 = col * PREVIEW_SCALE
                y0 = row * PREVIEW_SCALE
                self.preview.create_rectangle(
                    x0,
                    y0,
                    x0 + PREVIEW_SCALE,
                    y0 + PREVIEW_SCALE,
                    fill=color,
                    outline=color,
                )

    def _selected_model(self):
        label = self.model_selector.get()
        if not label or label not in self.model_options:
            raise RuntimeError("No model selected")

        source, value = self.model_options[label]
        key = (source, value)
        if self.model is not None and self.loaded_model_key == key:
            return self.model

        if source == "registry":
            model_info = get_model_info(value)
            self.model = load_registered_model(value)
            self.status_var.set(f"Loaded {model_info.run_id}")
        else:
            self.model = load_model(value)
            self.status_var.set(f"Loaded {value}")

        self.loaded_model_key = key
        return self.model

    def predict(self) -> None:
        try:
            model = self._selected_model()
            image = preprocess_digit_array(self.drawing)
            probabilities = model.predict(image, verbose=0)[0]
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error))
            return

        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])
        self.result_var.set(f"Prediction: {digit} ({confidence:.1%})")
        for index, probability in enumerate(probabilities):
            value = float(probability)
            self.probability_bars[index]["value"] = value
            self.probability_vars[index].set(f"{value:.1%}")

    def clear(self) -> None:
        self.canvas.delete("stroke")
        self.preview.delete("all")
        self.drawing.fill(0)
        self.result_var.set("Draw a digit, then predict.")
        for bar, value_var in zip(self.probability_bars, self.probability_vars):
            bar["value"] = 0.0
            value_var.set("0.0%")
        self.status_var.set("")

    def save_sample(self) -> None:
        label = simpledialog.askinteger(
            "Save sample",
            "Digit label (0-9):",
            parent=self.root,
            minvalue=0,
            maxvalue=9,
        )
        if label is None:
            return

        try:
            output_path = save_local_digit_sample(self.drawing, label, self.dataset_dir)
        except Exception as error:
            messagebox.showerror("Save failed", str(error))
            return

        self.status_var.set(f"Saved sample: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw and test MNIST-style digits locally.")
    parser.add_argument(
        "--model-id",
        default="latest",
        help="Registered model ID, or latest.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Fallback Keras model file if no registered model is available.",
    )
    parser.add_argument(
        "--local-dataset",
        default=str(LOCAL_DIGITS_DIR),
        help="Where saved labeled drawing samples are stored.",
    )
    return parser.parse_args()


def _load_runtime_dependencies() -> None:
    global cv2
    global get_model_info
    global list_models
    global load_model
    global load_registered_model
    global messagebox
    global np
    global prepare_mnist_style_image
    global preprocess_digit_array
    global save_local_digit_sample
    global simpledialog
    global tk
    global ttk

    import cv2 as cv2_module
    import numpy as np_module
    import tkinter as tk_module
    from tkinter import messagebox as messagebox_module
    from tkinter import simpledialog as simpledialog_module
    from tkinter import ttk as ttk_module

    from mnist_dnn.core import load_model as load_model_function
    from mnist_dnn.model_registry import (
        get_model_info as get_model_info_function,
        list_models as list_models_function,
        load_registered_model as load_registered_model_function,
    )
    from mnist_dnn.preprocessing import (
        prepare_mnist_style_image as prepare_mnist_style_image_function,
        preprocess_digit_array as preprocess_digit_array_function,
        save_local_digit_sample as save_local_digit_sample_function,
    )

    cv2 = cv2_module
    get_model_info = get_model_info_function
    list_models = list_models_function
    load_model = load_model_function
    load_registered_model = load_registered_model_function
    messagebox = messagebox_module
    np = np_module
    prepare_mnist_style_image = prepare_mnist_style_image_function
    preprocess_digit_array = preprocess_digit_array_function
    save_local_digit_sample = save_local_digit_sample_function
    simpledialog = simpledialog_module
    tk = tk_module
    ttk = ttk_module


def main() -> None:
    args = parse_args()
    _load_runtime_dependencies()
    root = tk.Tk()
    DigitTesterApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

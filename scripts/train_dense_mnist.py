"""Command-line entry point for the cleaned MNIST dense ANN prototype."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist_dnn import main  # noqa: E402


if __name__ == "__main__":
    main()

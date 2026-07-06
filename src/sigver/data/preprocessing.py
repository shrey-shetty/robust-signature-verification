"""Preprocessing helpers."""

from pathlib import Path


def preprocess_dataset(input_dir: str | Path, output_dir: str | Path) -> None:
    """Placeholder preprocessing entry point."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

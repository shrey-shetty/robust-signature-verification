"""Dataset wrappers for signature verification sources."""

from pathlib import Path
from typing import Optional


class SignatureDataset:
    """Simple placeholder dataset class for project scaffolding."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise NotImplementedError

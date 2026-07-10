"""Pair generation for Siamese (contrastive) training.

Definitions:
  - POSITIVE pair (label 1.0): two genuine signatures from the SAME writer.
  - NEGATIVE pair (label 0.0): either
      * skilled negative: a genuine signature and a skilled forgery
        targeting the same writer, or
      * random negative:  genuine signatures from two DIFFERENT writers
        (the "random forgery" scenario in the literature).

Label convention: 1.0 = same writer (similar), 0.0 = different (dissimilar).
The contrastive loss in sigver.losses must use the same convention.

Balance: for each writer, all genuine-genuine combinations are used as
positives (optionally capped), and an equal number of negatives is
sampled, split ~50/50 between skilled and random.

Everything is deterministic given the seed: pairs are index tuples into a
path-sorted sample list (see sigver.data.datasets.list_samples).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import torch
from torch.utils.data import Dataset

from sigver.data.datasets import Sample, SignatureDataset


@dataclass(frozen=True)
class Pair:
    i: int          # index into the sample list
    j: int
    label: float    # 1.0 same-writer genuine pair, 0.0 negative
    kind: str       # 'positive' | 'skilled' | 'random'


def generate_pairs(samples: list[Sample],
                   seed: int = 42,
                   max_positive_per_writer: int | None = None) -> list[Pair]:
    """Build a balanced, deterministic pair list from a sample list.

    max_positive_per_writer: cap on genuine-genuine combinations per
    writer (None = use all). With 24 genuine per writer, all combinations
    = 276 pairs/writer, which is fine at CEDAR/BHSig scale; set a cap for
    GPDS/institutional scale.
    """
    rng = random.Random(seed)

    genuine_by_writer: dict[int, list[int]] = defaultdict(list)
    forgery_by_writer: dict[int, list[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        (forgery_by_writer if s.is_forgery else genuine_by_writer)[s.writer_id].append(idx)

    writers = sorted(genuine_by_writer)
    if len(writers) < 2:
        raise ValueError("need at least 2 writers with genuine samples "
                         "to form random negatives")

    pairs: list[Pair] = []

    for w in writers:
        gen = genuine_by_writer[w]
        forg = forgery_by_writer.get(w, [])

        # --- positives: genuine-genuine combinations ---
        pos = list(combinations(gen, 2))
        rng.shuffle(pos)
        if max_positive_per_writer is not None:
            pos = pos[:max_positive_per_writer]
        for i, j in pos:
            pairs.append(Pair(i, j, 1.0, "positive"))

        n_neg = len(pos)
        n_skilled = n_neg // 2 if forg else 0
        n_random = n_neg - n_skilled

        # --- skilled negatives: genuine x own skilled forgery ---
        for _ in range(n_skilled):
            pairs.append(Pair(rng.choice(gen), rng.choice(forg), 0.0, "skilled"))

        # --- random negatives: genuine x other writer's genuine ---
        others = [ow for ow in writers if ow != w]
        for _ in range(n_random):
            ow = rng.choice(others)
            pairs.append(Pair(rng.choice(gen),
                              rng.choice(genuine_by_writer[ow]),
                              0.0, "random"))

    rng.shuffle(pairs)
    return pairs


def pair_summary(pairs: list[Pair]) -> dict[str, int]:
    counts = {"positive": 0, "skilled": 0, "random": 0}
    for p in pairs:
        counts[p.kind] += 1
    counts["total"] = len(pairs)
    return counts


class PairDataset(Dataset):
    """Yields (image_a, image_b, label) for contrastive training.

    Wraps a SignatureDataset (which handles preprocessing and optional
    in-memory caching) and a pair index list from generate_pairs.
    """

    def __init__(self, base: SignatureDataset, pairs: list[Pair]):
        self.base = base
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        img_a, _, _ = self.base[p.i]
        img_b, _, _ = self.base[p.j]
        label = torch.tensor(p.label, dtype=torch.float32)
        return img_a, img_b, label
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


@dataclass(frozen=True)
class Triplet:
    a: int          # anchor index
    p: int          # positive index (same writer, genuine)
    n: int          # negative index (skilled forgery or other writer)
    kind: str       # 'skilled' | 'random' -- which negative type was used


def generate_triplets(samples: list[Sample],
                      seed: int = 42,
                      max_positive_per_writer: int | None = None) -> list[Triplet]:
    """Build a balanced, deterministic triplet list for triplet-loss training.

    Mirrors generate_pairs' anchor/positive construction and skilled/random
    negative balance exactly, so the triplet and contrastive arms train on
    equivalent data and differ only in loss function:
      - anchor/positive come from the same combinations(gen, 2) construction,
        shuffled with the same random.Random(seed) discipline and subject to
        the same max_positive_per_writer cap
      - one triplet is emitted per anchor-positive pair; per writer, half use
        a skilled forgery of that writer as the negative and half use a
        genuine signature from a different writer (n_skilled = n_neg // 2,
        n_random = n_neg - n_skilled, matching generate_pairs)
      - a writer with no forgeries gets all-random negatives, exactly as
        generate_pairs does

    Deliberately no hard/semi-hard negative mining: the existing (contrastive)
    arm uses random sampling, so adding mining here would change a second
    variable and break the single-variable (loss-only) comparison this arm
    exists for. This is a scope decision, not an oversight.
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

    triplets: list[Triplet] = []

    for w in writers:
        gen = genuine_by_writer[w]
        forg = forgery_by_writer.get(w, [])

        # --- anchor/positive: genuine-genuine combinations ---
        pos = list(combinations(gen, 2))
        rng.shuffle(pos)
        if max_positive_per_writer is not None:
            pos = pos[:max_positive_per_writer]

        n_neg = len(pos)
        n_skilled = n_neg // 2 if forg else 0

        others = [ow for ow in writers if ow != w]
        for idx, (a, p) in enumerate(pos):
            if idx < n_skilled:
                triplets.append(Triplet(a, p, rng.choice(forg), "skilled"))
            else:
                ow = rng.choice(others)
                triplets.append(Triplet(a, p, rng.choice(genuine_by_writer[ow]), "random"))

    rng.shuffle(triplets)
    return triplets


def triplet_summary(triplets: list[Triplet]) -> dict[str, int]:
    counts = {"skilled": 0, "random": 0}
    for t in triplets:
        counts[t.kind] += 1
    counts["total"] = len(triplets)
    return counts


class TripletDataset(Dataset):
    """Yields (img_anchor, img_positive, img_negative) for triplet training.

    No label is returned: the argument ordering (anchor, positive, negative)
    encodes the same-writer / different-writer relationship that PairDataset
    instead carries in a separate label tensor.
    """

    def __init__(self, base: SignatureDataset, triplets: list[Triplet]):
        self.base = base
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        t = self.triplets[idx]
        img_a, _, _ = self.base[t.a]
        img_p, _, _ = self.base[t.p]
        img_n, _, _ = self.base[t.n]
        return img_a, img_p, img_n


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
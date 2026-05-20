"""
dataset.py — FaceRankNet
========================
PyTorch Dataset and DataLoader definitions.

Two dataset modes
-----------------
FaceDataset
    Standard single-face dataset.  Returns a face's 5 sub-graphs,
    its ground-truth holistic rating, its organ pseudo-scores, and
    (optionally) its ethnicity label.

PairDataset
    Wraps FaceDataset to yield (face_A, face_B, direction) triplets for
    the pairwise ranking loss.  For each anchor face A, samples one face B
    per organ where pseudo_score_A[organ] > pseudo_score_B[organ].

Reproducibility: np.random.seed(42) set at module level.
"""

from __future__ import annotations

import random

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
from organ_indices import ORGAN_INDICES
from preprocessing import build_all_subgraphs

np.random.seed(config.SEED)
random.seed(config.SEED)
torch.manual_seed(config.SEED)

# Organ name → integer index (consistent ordering)
ORGAN_ORDER: list[str] = config.ORGAN_NAMES


# ---------------------------------------------------------------------------
# FaceDataset
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    """
    Dataset for individual faces.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file with columns: Filename, Rating[, Ethnicity].
    coords_cache : dict[str, np.ndarray]
        Maps filename → (468, 3) normalised landmark array.
    pseudo_labels : dict[str, dict[str, float]]
        Maps filename → {organ: pseudo_score ∈ [1,5]}.
        Only required for training (pair sampling); pass {} for test.
    """

    def __init__(
        self,
        csv_path: str,
        coords_cache: dict[str, np.ndarray],
        pseudo_labels: dict[str, dict[str, float]] | None = None,
        avg_face: np.ndarray | None = None,
    ) -> None:
        df = pd.read_csv(csv_path)

        # Keep only rows with available landmarks
        mask = df[config.COL_FILENAME].isin(coords_cache)
        dropped = (~mask).sum()
        if dropped:
            import logging
            logging.getLogger(__name__).warning(
                "Dropped %d rows from CSV because landmarks are missing.", dropped
            )
        df = df[mask].reset_index(drop=True)

        self.filenames: list[str] = df[config.COL_FILENAME].tolist()
        self.ratings: list[float] = df[config.COL_RATING].astype(float).tolist()
        self.ethnicities: list[str | None] = (
            df[config.COL_ETHNICITY].tolist()
            if config.COL_ETHNICITY in df.columns
            else [None] * len(df)
        )
        self.coords_cache = coords_cache
        self.pseudo_labels = pseudo_labels or {}
        self.avg_face = avg_face  # None → 3-dim features; provided → 6-dim

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]
        coords = self.coords_cache[fname]                            # (468, 3)
        subgraphs = build_all_subgraphs(coords, self.avg_face)      # dict[str, DGLGraph]

        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)

        pseudo = self.pseudo_labels.get(fname, {})
        pseudo_tensor = torch.tensor(
            [pseudo.get(o, 3.0) for o in ORGAN_ORDER],
            dtype=torch.float32,
        )  # shape (5,)

        return {
            "filename": fname,
            "subgraphs": subgraphs,          # dict[str, DGLGraph]
            "rating": rating,                # scalar
            "pseudo_scores": pseudo_tensor,  # (5,)
            "ethnicity": self.ethnicities[idx],
        }


# ---------------------------------------------------------------------------
# PairDataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Wraps FaceDataset and yields triplets for the pairwise ranking loss.

    Each item is:
        (sample_A, sample_B, organ_mask)

    where ``organ_mask`` is a boolean tensor of shape (5,) that is True
    for organs where pseudo_score_A > pseudo_score_B.

    Only pairs for which at least one organ satisfies the ordering condition
    are returned.
    """

    def __init__(
        self,
        face_dataset: FaceDataset,
        pairs_per_sample: int = config.PAIRS_PER_SAMPLE,
    ) -> None:
        self.ds = face_dataset
        self.pairs_per_sample = pairs_per_sample

        # Pre-build list of valid (A_idx, B_idx) pairs
        self._pairs: list[tuple[int, int]] = self._build_pairs()

    def _build_pairs(self) -> list[tuple[int, int]]:
        n = len(self.ds)
        pairs: list[tuple[int, int]] = []
        indices = list(range(n))

        for a_idx in indices:
            pseudo_a = self.ds.pseudo_labels.get(self.ds.filenames[a_idx], {})
            if not pseudo_a:
                continue

            # Sample candidates
            candidates = random.sample(
                [i for i in indices if i != a_idx],
                k=min(self.pairs_per_sample * 10, n - 1),
            )

            added = 0
            for b_idx in candidates:
                pseudo_b = self.ds.pseudo_labels.get(
                    self.ds.filenames[b_idx], {}
                )
                if not pseudo_b:
                    continue

                # Valid if A scores higher than B in at least one organ
                dominates = any(
                    pseudo_a.get(o, 3.0) > pseudo_b.get(o, 3.0)
                    for o in ORGAN_ORDER
                )
                if dominates:
                    pairs.append((a_idx, b_idx))
                    added += 1
                    if added >= self.pairs_per_sample:
                        break

        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[dict, dict, torch.Tensor]:
        a_idx, b_idx = self._pairs[idx]
        sample_a = self.ds[a_idx]
        sample_b = self.ds[b_idx]

        pseudo_a = sample_a["pseudo_scores"]   # (5,)
        pseudo_b = sample_b["pseudo_scores"]   # (5,)
        organ_mask = (pseudo_a > pseudo_b)     # bool tensor (5,)

        return sample_a, sample_b, organ_mask


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_faces(
    batch: list[dict],
) -> dict:
    """
    Collate a list of FaceDataset items into a batched dict.

    DGL sub-graphs for the same organ are batched with ``dgl.batch()``.
    """
    organs = ORGAN_ORDER
    batched_subgraphs: dict[str, dgl.DGLGraph] = {
        o: dgl.batch([item["subgraphs"][o] for item in batch])
        for o in organs
    }
    return {
        "filenames": [item["filename"] for item in batch],
        "subgraphs": batched_subgraphs,
        "ratings": torch.stack([item["rating"] for item in batch]),
        "pseudo_scores": torch.stack([item["pseudo_scores"] for item in batch]),
        "ethnicities": [item["ethnicity"] for item in batch],
    }


def collate_pairs(
    batch: list[tuple[dict, dict, torch.Tensor]],
) -> tuple[dict, dict, torch.Tensor]:
    """Collate a list of PairDataset items."""
    batch_a = [item[0] for item in batch]
    batch_b = [item[1] for item in batch]
    masks = torch.stack([item[2] for item in batch])   # (B, 5)
    return collate_faces(batch_a), collate_faces(batch_b), masks


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_face_loader(
    dataset: FaceDataset,
    shuffle: bool = True,
    batch_size: int = config.BATCH_SIZE,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_faces,
    )


def make_pair_loader(
    pair_dataset: PairDataset,
    shuffle: bool = True,
    batch_size: int = config.BATCH_SIZE,
) -> DataLoader:
    return DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_pairs,
    )

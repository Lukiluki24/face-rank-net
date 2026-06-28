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
    Wraps FaceDataset to yield (face_A, face_B, organ_mask) triplets for
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from organ_indices import ORGAN_INDICES
from preprocessing import build_all_subgraphs

np.random.seed(config.SEED)
random.seed(config.SEED)
torch.manual_seed(config.SEED)

ORGAN_ORDER: list[str] = config.ORGAN_NAMES


# ---------------------------------------------------------------------------
# FaceDataset
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    """
    Dataset for individual faces.

    Parameters
    ----------
    csv_path : str | pd.DataFrame
        Path to a CSV file with columns: Filename, Rating[, Ethnicity],
        or a DataFrame directly.
    coords_cache : dict[str, np.ndarray]
        Maps filename → (468, 3) normalised landmark array.
    pseudo_labels : dict[str, dict[str, float]] | None
        Maps filename → {organ: pseudo_score ∈ [1,5]}.
        Only required for training (pair sampling); pass None for test.
    avg_face : np.ndarray | None
        (468, 3) universal average face. When provided, node features are
        6-dim (x, y, z, Δx, Δy, Δz); when None, 3-dim (x, y, z).
    augment_jitter : bool
        Add small Gaussian noise to landmark coords each __getitem__ call.
    """

    def __init__(
        self,
        csv_path: "str | pd.DataFrame",
        coords_cache: dict[str, np.ndarray],
        pseudo_labels: dict[str, dict[str, float]] | None = None,
        avg_face: np.ndarray | None = None,
        augment_jitter: bool = False,
        jitter_std: float = config.JITTER_STD,
    ) -> None:
        if isinstance(csv_path, pd.DataFrame):
            df = csv_path.reset_index(drop=True)
        else:
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
        self.avg_face = avg_face
        self.augment_jitter = augment_jitter
        self.jitter_std = jitter_std

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]
        coords = self.coords_cache[fname].copy()     # (468, 3)

        if self.augment_jitter and self.jitter_std > 0:
            coords = coords + np.random.normal(
                0.0, self.jitter_std, size=coords.shape
            ).astype(coords.dtype)

        subgraphs = build_all_subgraphs(coords, self.avg_face)

        pseudo = self.pseudo_labels.get(fname, {})
        pseudo_arr = np.array(
            [pseudo.get(o, 3.0) for o in ORGAN_ORDER], dtype=np.float32
        )

        return {
            "filename": fname,
            "subgraphs": subgraphs,
            "rating": torch.tensor(float(self.ratings[idx]), dtype=torch.float32),
            "pseudo_scores": torch.from_numpy(pseudo_arr),
            "ethnicity": self.ethnicities[idx],
        }


# ---------------------------------------------------------------------------
# PairDataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Wraps FaceDataset and yields triplets for the pairwise ranking loss.

    Each item is (sample_A, sample_B, organ_mask) where organ_mask is a
    boolean tensor of shape (5,) that is True for organs where
    pseudo_score_A > pseudo_score_B.
    """

    def __init__(
        self,
        face_dataset: FaceDataset,
        pairs_per_sample: int = config.PAIRS_PER_SAMPLE,
        hard_pair_sampling: bool = False,
    ) -> None:
        self.ds = face_dataset
        self.pairs_per_sample = pairs_per_sample
        self.hard_pair_sampling = hard_pair_sampling

        # Precompute bucket→indices map for Hard Pair Sampling
        self._bucket_indices: dict[int, list[int]] = {}
        for i, r in enumerate(self.ds.ratings):
            b = _bucket_of(r, config.BUCKET_EDGES)
            self._bucket_indices.setdefault(b, []).append(i)

        self._pairs: list[tuple[int, int]] = self._build_pairs()

    def _sample_candidates(self, a_idx: int, n: int, k: int) -> list[int]:
        """
        Choose candidate partner indices for anchor a_idx.

        With Hard Pair Sampling, samples more heavily from rating buckets
        far from the anchor's bucket.
        """
        if not self.hard_pair_sampling:
            return random.sample(
                [i for i in range(n) if i != a_idx],
                k=min(k, n - 1),
            )

        b_a = _bucket_of(self.ds.ratings[a_idx], config.BUCKET_EDGES)
        candidates: list[int] = []
        for b, members in self._bucket_indices.items():
            if not members:
                continue
            distance = abs(b - b_a)
            weight = distance + 1
            quota = max(1, int(round(k * weight / 10)))
            pool = [i for i in members if i != a_idx]
            if not pool:
                continue
            candidates.extend(random.sample(pool, k=min(quota, len(pool))))

        if len(candidates) > k:
            random.shuffle(candidates)
            candidates = candidates[:k]
        return candidates

    def _build_pairs(self) -> list[tuple[int, int]]:
        n = len(self.ds)
        pairs: list[tuple[int, int]] = []

        for a_idx in range(n):
            pseudo_a = self.ds.pseudo_labels.get(self.ds.filenames[a_idx], {})
            if not pseudo_a:
                continue

            candidates = self._sample_candidates(a_idx, n=n, k=self.pairs_per_sample * 10)

            rating_a = self.ds.ratings[a_idx]
            added = 0
            for b_idx in candidates:
                pseudo_b = self.ds.pseudo_labels.get(self.ds.filenames[b_idx], {})
                if not pseudo_b:
                    continue

                rating_b = self.ds.ratings[b_idx]
                if rating_a - rating_b < config.MIN_PAIR_RATING_GAP:
                    continue

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

        pseudo_a = sample_a["pseudo_scores"]
        pseudo_b = sample_b["pseudo_scores"]
        organ_mask = (pseudo_a - pseudo_b) > config.RANK_PSEUDO_MARGIN

        return sample_a, sample_b, organ_mask


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_faces(batch: list[dict]) -> dict:
    """Collate a list of FaceDataset items into a batched dict."""
    organs = ORGAN_ORDER
    batched_subgraphs: dict[str, dgl.DGLGraph] = {
        o: dgl.batch([item["subgraphs"][o] for item in batch])
        for o in organs
    }
    return {
        "filenames":   [item["filename"] for item in batch],
        "subgraphs":   batched_subgraphs,
        "ratings":     torch.stack([item["rating"] for item in batch]),
        "pseudo_scores": torch.stack([item["pseudo_scores"] for item in batch]),
        "ethnicities": [item["ethnicity"] for item in batch],
    }


def collate_pairs(
    batch: list[tuple[dict, dict, torch.Tensor]],
) -> tuple[dict, dict, torch.Tensor]:
    """Collate a list of PairDataset items."""
    batch_a = [item[0] for item in batch]
    batch_b = [item[1] for item in batch]
    masks = torch.stack([item[2] for item in batch])    # (B, 5)
    return collate_faces(batch_a), collate_faces(batch_b), masks


# ---------------------------------------------------------------------------
# Bucket helper
# ---------------------------------------------------------------------------

def _bucket_of(rating: float, edges: tuple[float, ...] = (2.0, 3.0, 4.0)) -> int:
    """Map a rating to a discrete bucket index (0..len(edges))."""
    b = 0
    for e in edges:
        if rating >= e:
            b += 1
    return b


# ---------------------------------------------------------------------------
# DataLoader factories
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


def make_weighted_pair_loader(
    pair_dataset: PairDataset,
    batch_size: int = config.BATCH_SIZE,
    bucket_edges: tuple[float, ...] = config.BUCKET_EDGES,
    smoothing: str = config.PAIR_SAMPLER_SMOOTHING,
) -> DataLoader:
    """
    Pair loader with WeightedRandomSampler that rebalances anchor (face A)
    rating buckets using config.BUCKET_EDGES.

    smoothing: "sqrt" (moderate boost) or "inverse" (aggressive boost).
    """
    ratings = pair_dataset.ds.ratings
    pair_buckets = np.array(
        [_bucket_of(ratings[a_idx], bucket_edges) for (a_idx, _) in pair_dataset._pairs],
        dtype=np.int64,
    )
    n_buckets = len(bucket_edges) + 1
    counts = np.bincount(pair_buckets, minlength=n_buckets).astype(np.float64)
    safe = np.maximum(counts, 1.0)
    if smoothing == "inverse":
        inv = 1.0 / safe
    elif smoothing == "sqrt":
        inv = 1.0 / np.sqrt(safe)
    else:
        raise ValueError(f"Unknown smoothing mode: {smoothing!r}")
    weights = inv[pair_buckets]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )

    return DataLoader(
        pair_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_pairs,
    )

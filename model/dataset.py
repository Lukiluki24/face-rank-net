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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from organ_indices import ORGAN_INDICES
from preprocessing import build_all_subgraphs, build_all_subgraphs_flipped

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
        csv_path: "str | pd.DataFrame",
        coords_cache: dict[str, np.ndarray],
        pseudo_labels: dict[str, dict[str, float]] | None = None,
        avg_face: np.ndarray | None = None,
        augment_flip: bool = False,
        augment_jitter: bool = False,
        jitter_std: float = config.JITTER_STD,
        compute_lds_weights: bool = False,
        mixup_within_bucket: bool = False,
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

        base_filenames: list[str] = df[config.COL_FILENAME].tolist()
        base_ratings: list[float] = df[config.COL_RATING].astype(float).tolist()
        base_ethnicities: list[str | None] = (
            df[config.COL_ETHNICITY].tolist()
            if config.COL_ETHNICITY in df.columns
            else [None] * len(df)
        )

        if augment_flip:
            # Duplicate dataset: original + mirrored copies
            self.filenames = base_filenames + base_filenames
            self.ratings   = base_ratings   + base_ratings
            self.ethnicities = base_ethnicities + base_ethnicities
            self._is_flipped: list[bool] = (
                [False] * len(base_filenames) + [True] * len(base_filenames)
            )
        else:
            self.filenames   = base_filenames
            self.ratings     = base_ratings
            self.ethnicities = base_ethnicities
            self._is_flipped = [False] * len(base_filenames)

        self.coords_cache = coords_cache
        self.pseudo_labels = pseudo_labels or {}
        self.avg_face = avg_face  # None → 3-dim features; provided → 6-dim
        self.augment_jitter = augment_jitter
        self.jitter_std = jitter_std

        # ---- LDS (Label Distribution Smoothing) per-sample weights ----
        # Used by l_reg(weights=...) to penalise rare-rating errors more.
        # Precomputed once on train ratings; test set should pass False.
        if compute_lds_weights:
            self._rating_weights = self._build_lds_weights(self.ratings)
        else:
            self._rating_weights = [1.0] * len(self.filenames)

        # ---- Within-bucket MixUp setup ----
        self.mixup_within_bucket = mixup_within_bucket
        edges = config.MIXUP_BUCKET_EDGES
        self._bucket_assignment = [_bucket_of(r, edges) for r in self.ratings]
        self._bucket_to_indices: dict[int, list[int]] = {}
        for i, b in enumerate(self._bucket_assignment):
            self._bucket_to_indices.setdefault(b, []).append(i)
        if self.mixup_within_bucket:
            import logging
            log = logging.getLogger(__name__)
            log.info(
                "Within-bucket MixUp enabled (prob=%.2f, α=%.2f, tail-only=%s).",
                config.MIXUP_PROB, config.MIXUP_ALPHA, config.MIXUP_TAIL_BUCKETS_ONLY,
            )

    @staticmethod
    def _build_lds_weights(ratings: list[float]) -> list[float]:
        """
        Compute per-sample inverse-density weights via KDE over rating
        distribution. Follows Yang et al. 2021 (LDS, ICML).
        """
        from scipy.stats import gaussian_kde

        arr = np.asarray(ratings, dtype=np.float64)
        kde = gaussian_kde(arr, bw_method=config.LREG_LDS_BANDWIDTH)
        density = kde(arr)
        # Inverse density, floor to prevent runaway weights at extreme tails
        inv = 1.0 / np.clip(density, config.LREG_WEIGHT_FLOOR, None)
        # Normalise to mean 1 so the loss scale stays comparable to plain MSE
        inv = inv * (len(inv) / inv.sum())
        return inv.tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]
        coords = self.coords_cache[fname]                            # (468, 3)
        rating_val = float(self.ratings[idx])
        weight_val = float(self._rating_weights[idx])

        pseudo = self.pseudo_labels.get(fname, {})
        pseudo_arr = np.array(
            [pseudo.get(o, 3.0) for o in ORGAN_ORDER], dtype=np.float32
        )

        # ---- Within-bucket MixUp ----
        # When enabled, sometimes mix this face with another face from the
        # SAME rating bucket: extra tail diversity (Jelek↔Jelek, Cantik↔
        # Cantik). Skipped for flipped duplicates so we don't double-mix.
        if (
            self.mixup_within_bucket
            and not self._is_flipped[idx]
            and np.random.rand() < config.MIXUP_PROB
        ):
            my_bucket = self._bucket_assignment[idx]
            tail_only = config.MIXUP_TAIL_BUCKETS_ONLY
            n_buckets = len(config.MIXUP_BUCKET_EDGES) + 1
            is_tail = my_bucket == 0 or my_bucket == n_buckets - 1
            if (not tail_only or is_tail):
                pool = [i for i in self._bucket_to_indices.get(my_bucket, [])
                        if i != idx and not self._is_flipped[i]]
                if pool:
                    partner_idx = pool[np.random.randint(len(pool))]
                    α = float(np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA))
                    partner_coords = self.coords_cache[self.filenames[partner_idx]]
                    coords = (α * coords + (1.0 - α) * partner_coords).astype(coords.dtype)
                    rating_val = α * rating_val + (1.0 - α) * float(self.ratings[partner_idx])
                    partner_pseudo = self.pseudo_labels.get(
                        self.filenames[partner_idx], {}
                    )
                    partner_arr = np.array(
                        [partner_pseudo.get(o, 3.0) for o in ORGAN_ORDER],
                        dtype=np.float32,
                    )
                    pseudo_arr = α * pseudo_arr + (1.0 - α) * partner_arr
                    weight_val = α * weight_val + (1.0 - α) * float(
                        self._rating_weights[partner_idx]
                    )

        # Landmark jitter: tiny Gaussian noise per call → minority faces
        # resampled by WeightedRandomSampler see slightly different geometry
        # each time, preventing overfitting to a small unique set.
        if self.augment_jitter and self.jitter_std > 0:
            coords = coords + np.random.normal(
                0.0, self.jitter_std, size=coords.shape
            ).astype(coords.dtype)

        if self._is_flipped[idx]:
            subgraphs = build_all_subgraphs_flipped(coords, self.avg_face)
        else:
            subgraphs = build_all_subgraphs(coords, self.avg_face)  # dict[str, DGLGraph]

        return {
            "filename": fname,
            "subgraphs": subgraphs,          # dict[str, DGLGraph]
            "rating": torch.tensor(rating_val, dtype=torch.float32),
            "pseudo_scores": torch.from_numpy(pseudo_arr),           # (5,)
            "ethnicity": self.ethnicities[idx],
            "rating_weight": torch.tensor(weight_val, dtype=torch.float32),
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
        hard_pair_sampling: bool = False,
    ) -> None:
        self.ds = face_dataset
        self.pairs_per_sample = pairs_per_sample
        self.hard_pair_sampling = hard_pair_sampling

        # Precompute bucket→indices map once for Hard Pair Sampling
        bucket_edges = config.MIXUP_BUCKET_EDGES
        self._bucket_indices: dict[int, list[int]] = {}
        for i, r in enumerate(self.ds.ratings):
            b = _bucket_of(r, bucket_edges)
            self._bucket_indices.setdefault(b, []).append(i)

        # Pre-build list of valid (A_idx, B_idx) pairs
        self._pairs: list[tuple[int, int]] = self._build_pairs()

    def _sample_candidates(
        self,
        a_idx: int,
        n: int,
        k: int,
    ) -> list[int]:
        """
        Choose candidate partner indices for anchor a_idx.

        With Hard Pair Sampling enabled, sample more heavily from rating
        buckets that are FAR from the anchor's bucket. With random pairing
        (default) the chance of seeing a Jelek↔Cantik pair is ~0.5 %; HPS
        boosts this dramatically so L_rank actually learns extreme contrasts.
        """
        if not self.hard_pair_sampling:
            return random.sample(
                [i for i in range(n) if i != a_idx],
                k=min(k, n - 1),
            )

        edges = config.MIXUP_BUCKET_EDGES
        b_a = _bucket_of(self.ds.ratings[a_idx], edges)

        # Stratified pool: more slots for distant buckets
        candidates: list[int] = []
        for b, members in self._bucket_indices.items():
            if not members:
                continue
            # Distance 0 → small quota; large distance → large quota.
            distance = abs(b - b_a)
            weight = distance + 1  # 1, 2, 3, 4 for distances 0..3
            quota = max(1, int(round(k * weight / 10)))
            pool = [i for i in members if i != a_idx]
            if not pool:
                continue
            quota = min(quota, len(pool))
            candidates.extend(random.sample(pool, k=quota))

        if len(candidates) > k:
            random.shuffle(candidates)
            candidates = candidates[:k]
        return candidates

    def _build_pairs(self) -> list[tuple[int, int]]:
        n = len(self.ds)
        pairs: list[tuple[int, int]] = []
        indices = list(range(n))

        for a_idx in indices:
            pseudo_a = self.ds.pseudo_labels.get(self.ds.filenames[a_idx], {})
            if not pseudo_a:
                continue

            candidates = self._sample_candidates(
                a_idx, n=n, k=self.pairs_per_sample * 10
            )

            rating_a = self.ds.ratings[a_idx]
            added = 0
            for b_idx in candidates:
                pseudo_b = self.ds.pseudo_labels.get(
                    self.ds.filenames[b_idx], {}
                )
                if not pseudo_b:
                    continue

                # H2: only train L_rank when holistic order agrees with pseudo order.
                # MIN_PAIR_RATING_GAP tightens this beyond simple > to skip near-tie
                # pairs whose L_rank signal is too noisy to be useful.
                rating_b = self.ds.ratings[b_idx]
                if rating_a - rating_b < config.MIN_PAIR_RATING_GAP:
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
        # Confidence margin: pseudo-labels are noisy (ρ≈0.57); ignore organs
        # where the gap is below the noise floor so L_rank trains only on
        # high-confidence orderings.
        organ_mask = (pseudo_a - pseudo_b) > config.RANK_PSEUDO_MARGIN

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
        "rating_weights": torch.stack([
            item.get("rating_weight", torch.tensor(1.0)) for item in batch
        ]),
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


def _bucket_of(rating: float, edges: tuple[float, ...] = (2.0, 3.0, 4.0)) -> int:
    """Map a rating to a discrete bucket index (0..len(edges))."""
    b = 0
    for e in edges:
        if rating >= e:
            b += 1
    return b


def make_weighted_pair_loader(
    pair_dataset: PairDataset,
    batch_size: int = config.BATCH_SIZE,
    bucket_edges: tuple[float, ...] = (2.0, 3.0, 4.0),
    smoothing: str = "sqrt",
) -> DataLoader:
    """
    Pair loader with WeightedRandomSampler that rebalances anchor (face A)
    rating buckets.

    SCUT-FBP5500 is heavily imbalanced:
        Jelek (<2)  ~ 4.7%  (~188 unique)  ←  rare extremes
        2–3         ~55%
        3–4         ~29%
        Cantik (>4) ~11%    (~482 unique)  ←  rare extremes

    Smoothing modes
    ---------------
    "inverse" — weight ∝ 1 / count   (strong rebalance; ~13× boost on Jelek)
                  Risk: overfits minority class when unique count < ~200.
    "sqrt"    — weight ∝ 1 / √count  (moderate rebalance; ~3.6× boost on Jelek)
                  Recommended default — keeps minority signal up without
                  forcing the model to re-see the same 188 faces too often.

    Notes
    -----
    Pair list is precomputed in PairDataset; the sampler picks pair indices
    (not face indices), so this rebalances the **distribution of pairs seen
    per epoch** without altering the H2-validated pair pool itself.
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

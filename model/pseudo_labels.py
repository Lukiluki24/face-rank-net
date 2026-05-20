"""
pseudo_labels.py — FaceRankNet
================================
Implements the *Averageness Hypothesis* for organ-level pseudo-label generation.

Pipeline
--------
1. compute_universal_average_face  — element-wise mean of all train coords.
2. compute_organ_mse               — MSE between one face's organ and avg face.
3. compute_organ_pseudo_score      — linearly map MSE → score ∈ [1, 5].
4. compute_all_pseudo_labels       — run over the whole training set.
5. save / load helpers             — pickle-based caching.

Averageness Hypothesis:
    Faces closer to the universal average face are perceived as more attractive.
    Therefore lower MSE → higher beauty score.

Score mapping (from instruction):
    score = 5 - 4 * (mse / max_mse_across_dataset)
    Clipped to [1, 5].
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from organ_indices import ORGAN_INDICES

np.random.seed(config.SEED)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Universal Average Face
# ---------------------------------------------------------------------------

def compute_universal_average_face(
    coords_list: list[np.ndarray],
) -> np.ndarray:
    """
    Compute the dataset-wide Universal Average Face.

    Parameters
    ----------
    coords_list : list[np.ndarray]
        Each element is a (468, 3) float32 normalised coordinate array
        from the **training** set.

    Returns
    -------
    np.ndarray
        Shape (468, 3) float32 — element-wise mean across all faces.
    """
    if not coords_list:
        raise ValueError("coords_list must be non-empty.")

    stack = np.stack(coords_list, axis=0)   # (N_faces, 468, 3)
    avg_face = stack.mean(axis=0)            # (468, 3)
    logger.info(
        "Universal Average Face computed from %d faces.", len(coords_list)
    )
    return avg_face.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-organ MSE helpers
# ---------------------------------------------------------------------------

def compute_organ_mse(
    coords: np.ndarray,
    avg_face: np.ndarray,
    organ_indices: list[int],
) -> float:
    """
    Mean squared error between this face's organ nodes and the avg face.

    Parameters
    ----------
    coords : np.ndarray
        Shape (468, 3) — normalised landmark coordinates for a single face.
    avg_face : np.ndarray
        Shape (468, 3) — Universal Average Face.
    organ_indices : list[int]
        MediaPipe landmark indices for the organ of interest.

    Returns
    -------
    float
        MSE (non-negative scalar).
    """
    diff = coords[organ_indices] - avg_face[organ_indices]  # (N_nodes, 3)
    return float(np.sqrt(np.mean(diff ** 2)))  # RMSE: unit-consistent across organs


def compute_organ_pseudo_score(
    coords: np.ndarray,
    avg_face: np.ndarray,
    organ_indices: list[int],
    max_mse: float,
) -> float:
    """
    Map organ MSE to a beauty pseudo-score in [1, 5].

    Formula (from instruction):
        score = 5 - 4 × (mse / max_mse_across_dataset)

    Parameters
    ----------
    coords : np.ndarray
        Shape (468, 3) — normalised face coordinates.
    avg_face : np.ndarray
        Shape (468, 3) — Universal Average Face.
    organ_indices : list[int]
        Landmark indices for this organ.
    max_mse : float
        Maximum MSE observed across the dataset for this organ
        (used for linear normalisation).

    Returns
    -------
    float
        Beauty pseudo-score in [1.0, 5.0].
    """
    mse = compute_organ_mse(coords, avg_face, organ_indices)
    if max_mse < 1e-12:
        return 5.0  # degenerate: every face is the average
    score = 5.0 - 4.0 * (mse / max_mse)
    return float(np.clip(score, 1.0, 5.0))


# ---------------------------------------------------------------------------
# Dataset-level pseudo-label computation
# ---------------------------------------------------------------------------

def compute_all_pseudo_labels(
    coords_cache: dict[str, np.ndarray],
    avg_face: np.ndarray,
    train_filenames: list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute pseudo-labels for every image in the training set.

    Algorithm
    ---------
    1. For each organ, collect all MSE values across the training set.
    2. Record ``max_mse`` per organ.
    3. Map every face's organ MSE to [1, 5] using ``max_mse``.

    Parameters
    ----------
    coords_cache : dict[str, np.ndarray]
        Maps filename → (468, 3) normalised coords.
    avg_face : np.ndarray
        Shape (468, 3) — Universal Average Face (computed from training set).
    train_filenames : list[str]
        Ordered list of training-set filenames (only these are processed).

    Returns
    -------
    dict[str, dict[str, float]]
        Maps filename → {organ_name: pseudo_score}, all scores in [1, 5].
    """
    # ---- Pass 1: collect MSE per organ ----
    organ_mse_all: dict[str, list[float]] = {o: [] for o in ORGAN_INDICES}
    valid_fnames = [f for f in train_filenames if f in coords_cache]

    for fname in tqdm(valid_fnames, desc="Pass 1 — collecting MSEs", unit="face"):
        coords = coords_cache[fname]
        for organ, idxs in ORGAN_INDICES.items():
            organ_mse_all[organ].append(
                compute_organ_mse(coords, avg_face, idxs)
            )

    max_mse_per_organ: dict[str, float] = {
        organ: float(max(vals)) if vals else 1.0
        for organ, vals in organ_mse_all.items()
    }
    logger.info("Max MSE per organ: %s", max_mse_per_organ)

    # ---- Pass 2: compute pseudo scores ----
    pseudo_labels: dict[str, dict[str, float]] = {}

    for fname in tqdm(valid_fnames, desc="Pass 2 — pseudo scores", unit="face"):
        coords = coords_cache[fname]
        pseudo_labels[fname] = {
            organ: compute_organ_pseudo_score(
                coords, avg_face, idxs, max_mse_per_organ[organ]
            )
            for organ, idxs in ORGAN_INDICES.items()
        }

    logger.info(
        "Pseudo-labels computed for %d / %d training images.",
        len(pseudo_labels),
        len(train_filenames),
    )
    return pseudo_labels


# ---------------------------------------------------------------------------
# Pseudo-label quality diagnostic
# ---------------------------------------------------------------------------

def validate_pseudo_label_quality(
    pseudo_labels: dict[str, dict[str, float]],
    holistic_ratings: dict[str, float],
) -> float:
    """
    Compute Spearman ρ between mean organ pseudo-score and holistic rating.

    A low ρ (< 0.2) indicates the pseudo-labels are weakly correlated with
    actual beauty scores — meaning L_rank may conflict with L_reg during training.

    Parameters
    ----------
    pseudo_labels : dict[str, dict[str, float]]
        Maps filename → {organ: pseudo_score}.
    holistic_ratings : dict[str, float]
        Maps filename → ground-truth holistic beauty score.

    Returns
    -------
    float
        Spearman ρ ∈ [-1, 1]. Prints a warning if ρ < 0.2.
    """
    from scipy.stats import spearmanr

    common = [f for f in pseudo_labels if f in holistic_ratings]
    if len(common) < 10:
        logger.warning("Too few common samples (%d) to compute Spearman ρ.", len(common))
        return float("nan")

    mean_pseudo = [float(np.mean(list(pseudo_labels[f].values()))) for f in common]
    ratings     = [holistic_ratings[f] for f in common]

    rho, p_val = spearmanr(mean_pseudo, ratings)
    level = "✓ GOOD" if rho >= 0.3 else ("~ WEAK" if rho >= 0.1 else "✗ POOR")
    logger.info(
        "Pseudo-label quality — Spearman ρ = %.4f  (p=%.4f)  %s  [n=%d]",
        rho, p_val, level, len(common),
    )
    if rho < 0.2:
        logger.warning(
            "Spearman ρ = %.4f < 0.2 — pseudo-labels are weakly aligned with "
            "holistic ratings. L_rank may conflict with L_reg during training.",
            rho,
        )
    return float(rho)


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_pseudo_labels(
    pseudo_labels: dict[str, dict[str, float]],
    cache_path: str,
) -> None:
    """Persist pseudo-labels dict to a pickle file."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(pseudo_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Pseudo-labels saved to '%s'.", p)


def load_pseudo_labels(
    cache_path: str,
) -> dict[str, dict[str, float]]:
    """Load pseudo-labels dict from a pickle file."""
    with open(cache_path, "rb") as f:
        data: dict[str, dict[str, float]] = pickle.load(f)
    logger.info("Loaded pseudo-labels for %d faces from '%s'.", len(data), cache_path)
    return data


def save_avg_face(avg_face: np.ndarray, cache_path: str) -> None:
    """Persist Universal Average Face to a .npy file."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), avg_face)
    logger.info("Average face saved to '%s'.", p)


def load_avg_face(cache_path: str) -> np.ndarray:
    """Load Universal Average Face from a .npy file."""
    avg_face = np.load(cache_path).astype(np.float32)
    logger.info("Average face loaded from '%s', shape=%s.", cache_path, avg_face.shape)
    return avg_face


# ---------------------------------------------------------------------------
# CLI entry-point  (Colab Cell 4: %run pseudo_labels.py ...)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pickle

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FaceRankNet — compute Universal Average Face & pseudo-labels"
    )
    parser.add_argument("--landmark_cache", required=True,
                        help="Path to train landmark .pkl from preprocessing.py")
    parser.add_argument("--train_csv", required=True,
                        help="CSV with 'Filename' column for the training set")
    parser.add_argument("--avg_face_out", default=str(config.AVG_FACE_CACHE),
                        help="Output path for avg_face.npy")
    parser.add_argument("--pseudo_labels_out", default=str(config.PSEUDO_LABEL_CACHE),
                        help="Output path for pseudo_labels.pkl")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.train_csv)
    train_filenames: list[str] = df[config.COL_FILENAME].tolist()

    with open(args.landmark_cache, "rb") as fh:
        coords_cache: dict[str, np.ndarray] = pickle.load(fh)

    train_coords = [
        coords_cache[f] for f in train_filenames if f in coords_cache
    ]

    avg_face = compute_universal_average_face(train_coords)
    save_avg_face(avg_face, args.avg_face_out)

    pseudo_labels = compute_all_pseudo_labels(
        coords_cache, avg_face, train_filenames
    )
    save_pseudo_labels(pseudo_labels, args.pseudo_labels_out)

"""
pseudo_labels.py — FaceRankNet
================================
Implements the *Averageness Hypothesis* for organ-level pseudo-label generation.

Pipeline
--------
1. compute_universal_average_face  — element-wise mean of all train coords.
2. compute_beauty_prototype        — mean of top-k% highest-rated faces (beauty ideal).
3. compute_organ_mse               — RMSE between one face's organ and reference face.
4. compute_organ_pseudo_score      — linearly map MSE → score ∈ [1, 5].
5. compute_all_pseudo_labels       — run over the whole training set.
6. save / load helpers             — pickle-based caching.

Averageness Hypothesis (refined):
    Faces closer to the *beauty prototype* (avg of top-rated faces) are more
    attractive.  Using a population-wide average mixes attractive and unattractive
    faces, weakening the signal.  The beauty prototype isolates the attractive
    subspace, so lower RMSE → higher pseudo-score aligns with holistic ratings.

Score mapping:
    Percentile-based: rank face by RMSE within dataset → map to [1, 5].
    Lower RMSE (closer to prototype) → lower rank → score closer to 5.
    Avoids compression caused by a single outlier driving max_RMSE.
"""

from __future__ import annotations

import bisect
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

def compute_beauty_prototype(
    coords_list: list[np.ndarray],
    ratings: list[float],
    top_k_pct: float = 0.30,
) -> np.ndarray:
    """
    Compute the Beauty Prototype: mean face of the top-k% highest-rated faces.

    Using population average mixes attractive + unattractive faces, weakening
    the averageness signal.  The beauty prototype isolates the attractive
    subspace so that RMSE(face, prototype) inversely predicts beauty.

    Parameters
    ----------
    coords_list : list[np.ndarray]
        Each element is a (468, 3) float32 normalised coordinate array.
    ratings : list[float]
        Holistic beauty ratings corresponding to each entry in coords_list.
    top_k_pct : float
        Fraction of top-rated faces to include (default 0.30 = top 30%).

    Returns
    -------
    np.ndarray
        Shape (468, 3) float32 — mean of top-k% faces.
    """
    if not coords_list:
        raise ValueError("coords_list must be non-empty.")

    n_top = max(1, int(len(ratings) * top_k_pct))
    sorted_indices = np.argsort(ratings)[::-1][:n_top]
    top_coords = [coords_list[i] for i in sorted_indices]

    prototype = np.stack(top_coords, axis=0).mean(axis=0)
    logger.info(
        "Beauty prototype computed from top %d / %d faces (top %.0f%%).",
        n_top, len(coords_list), top_k_pct * 100,
    )
    return prototype.astype(np.float32)


def compute_ethnicity_avg_faces(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    ethnicity_map: dict[str, str],
    holistic_ratings: dict[str, float] | None = None,
    top_k_pct: float = 0.30,
) -> dict[str, np.ndarray]:
    """
    Compute per-ethnicity Beauty Prototypes (H1 refinement).

    If holistic_ratings is provided, uses compute_beauty_prototype (top-k%).
    Otherwise falls back to population average (compute_universal_average_face).

    Returns dict mapping ethnicity string → (468, 3) prototype array.
    """
    groups_coords: dict[str, list[np.ndarray]] = {}
    groups_ratings: dict[str, list[float]] = {}

    for fname in train_filenames:
        if fname not in coords_cache:
            continue
        eth = ethnicity_map.get(fname, "Unknown")
        groups_coords.setdefault(eth, []).append(coords_cache[fname])
        if holistic_ratings is not None:
            groups_ratings.setdefault(eth, []).append(
                holistic_ratings.get(fname, 3.0)
            )

    result = {}
    for eth, coords_list in groups_coords.items():
        if holistic_ratings is not None and eth in groups_ratings:
            result[eth] = compute_beauty_prototype(
                coords_list, groups_ratings[eth], top_k_pct=top_k_pct
            )
        else:
            result[eth] = compute_universal_average_face(coords_list)
        logger.info(
            "Reference face for '%s' computed from %d faces.", eth, len(coords_list)
        )
    return result


def compute_all_pseudo_labels(
    coords_cache: dict[str, np.ndarray],
    avg_face: np.ndarray,
    train_filenames: list[str],
    avg_face_map: dict[str, np.ndarray] | None = None,
    ethnicity_map: dict[str, str] | None = None,
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

    use_ethnicity = avg_face_map is not None and ethnicity_map is not None

    for fname in tqdm(valid_fnames, desc="Pass 1 — collecting MSEs", unit="face"):
        coords = coords_cache[fname]
        face_avg = avg_face
        if use_ethnicity:
            eth = ethnicity_map.get(fname, "Unknown")
            face_avg = avg_face_map.get(eth, avg_face)
        for organ, idxs in ORGAN_INDICES.items():
            organ_mse_all[organ].append(
                compute_organ_mse(coords, face_avg, idxs)
            )

    # Percentile-based normalization: sort each organ's MSE list so rank lookup
    # is O(log n). Avoids compression caused by a single outlier driving max_MSE.
    organ_sorted_mse: dict[str, list[float]] = {
        organ: sorted(vals) for organ, vals in organ_mse_all.items()
    }
    logger.info(
        "Organ MSE ranges (min→max): %s",
        {o: (round(v[0], 5), round(v[-1], 5)) for o, v in organ_sorted_mse.items()},
    )

    # ---- Pass 2: compute pseudo scores via percentile rank ----
    # Direction: lower RMSE from beauty prototype → higher pseudo-score.
    # Percentile-based avoids compression from a single max-RMSE outlier,
    # ensuring scores are uniformly spread across [1, 5].
    pseudo_labels: dict[str, dict[str, float]] = {}

    for fname in tqdm(valid_fnames, desc="Pass 2 — pseudo scores", unit="face"):
        coords = coords_cache[fname]
        face_avg = avg_face
        if use_ethnicity:
            eth = ethnicity_map.get(fname, "Unknown")
            face_avg = avg_face_map.get(eth, avg_face)

        scores: dict[str, float] = {}
        for organ, idxs in ORGAN_INDICES.items():
            mse = compute_organ_mse(coords, face_avg, idxs)
            sorted_vals = organ_sorted_mse[organ]
            n = len(sorted_vals)
            # Percentile rank ∈ [0, 1): 0 = lowest RMSE, 1 = highest RMSE
            rank = bisect.bisect_left(sorted_vals, mse) / n
            # Averageness: lower RMSE (rank near 0) → score near 5
            score = float(np.clip(5.0 - 4.0 * rank, 1.0, 5.0))
            scores[organ] = score
        pseudo_labels[fname] = scores

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

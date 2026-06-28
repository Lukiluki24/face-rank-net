"""
pseudo_labels.py — FaceRankNet
================================
Two pseudo-label generation methods based on the *Averageness Hypothesis*:

  1. compute_all_pseudo_labels              (classic RMSE-from-prototype)
  2. compute_all_pseudo_labels_beauty_axis  (beauty-direction projection)

Pipeline
--------
1. compute_universal_average_face  — element-wise mean of all train coords.
2. compute_beauty_prototype        — mean of top-k% highest-rated faces.
3. compute_organ_mse               — RMSE between a face's organ and reference.
4. compute_all_pseudo_labels       — run over the whole training set.
5. save / load helpers             — pickle-based caching.

Averageness Hypothesis (refined):
    Faces closer to the *beauty prototype* (avg of top-rated faces) are more
    attractive. The prototype isolates the attractive subspace, so lower
    RMSE → higher pseudo-score aligns with holistic ratings.

Beauty-axis variant:
    Following Said & Todorov (2011), attractiveness is better modelled as a
    *direction* in face space than as proximity to a fixed prototype. Faces
    that deviate from the population mean *toward* the beauty direction get
    high scores even when they sit far from the prototype overall.
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
    """Element-wise mean of all (468, 3) coord arrays."""
    if not coords_list:
        raise ValueError("coords_list must be non-empty.")

    stack = np.stack(coords_list, axis=0)
    avg_face = stack.mean(axis=0)
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
    """RMSE between this face's organ nodes and the reference face."""
    diff = coords[organ_indices] - avg_face[organ_indices]
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Beauty prototype + axis
# ---------------------------------------------------------------------------

def compute_beauty_prototype(
    coords_list: list[np.ndarray],
    ratings: list[float],
    top_k_pct: float = 0.30,
) -> np.ndarray:
    """
    Beauty Prototype: mean face of the top-k% highest-rated faces.

    Using a population average mixes attractive + unattractive faces, weakening
    the averageness signal. The prototype isolates the attractive subspace.
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


def compute_beauty_axis(
    population_mean: np.ndarray,
    beauty_prototype: np.ndarray,
) -> np.ndarray:
    """
    Beauty axis = direction from population mean → beauty prototype.

    Following Said & Todorov (2011), attractiveness is better modelled as a
    direction in face space than as proximity to a fixed prototype.
    """
    return (beauty_prototype - population_mean).astype(np.float32)


def project_organ_onto_axis(
    coords: np.ndarray,
    population_mean: np.ndarray,
    beauty_axis: np.ndarray,
    organ_indices: list[int],
) -> float:
    """
    Scalar projection of an organ's geometric deviation onto the beauty axis.

    Positive  → organ deviates from the mean *toward* the beauty direction.
    Negative  → opposite of beauty direction (less attractive).
    """
    deviation = coords[organ_indices] - population_mean[organ_indices]
    axis      = beauty_axis[organ_indices]

    dev_flat  = deviation.ravel()
    axis_flat = axis.ravel()

    axis_norm = float(np.linalg.norm(axis_flat))
    if axis_norm < 1e-12:
        return 0.0
    return float(np.dot(dev_flat, axis_flat) / axis_norm)


# ---------------------------------------------------------------------------
# Per-ethnicity reference faces (H1 refinement)
# ---------------------------------------------------------------------------

def compute_ethnicity_avg_faces(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    ethnicity_map: dict[str, str],
    holistic_ratings: dict[str, float] | None = None,
    top_k_pct: float = 0.30,
) -> dict[str, np.ndarray]:
    """
    Compute per-ethnicity reference faces (H1 refinement).

    If holistic_ratings is provided, uses Beauty Prototype (top-k%).
    Otherwise falls back to population average.
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


# ---------------------------------------------------------------------------
# Method 1: classic RMSE-from-prototype pseudo-labels
# ---------------------------------------------------------------------------

def compute_all_pseudo_labels(
    coords_cache: dict[str, np.ndarray],
    avg_face: np.ndarray,
    train_filenames: list[str],
    avg_face_map: dict[str, np.ndarray] | None = None,
    ethnicity_map: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Classic Averageness pseudo-labels via percentile-ranked organ RMSE.

    Lower RMSE from the (per-ethnicity) reference face → higher pseudo-score.
    Percentile-based normalization avoids compression from single outliers.
    """
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

    organ_sorted_mse: dict[str, list[float]] = {
        organ: sorted(vals) for organ, vals in organ_mse_all.items()
    }
    logger.info(
        "Organ MSE ranges (min→max): %s",
        {o: (round(v[0], 5), round(v[-1], 5)) for o, v in organ_sorted_mse.items()},
    )

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
            rank = bisect.bisect_left(sorted_vals, mse) / n
            score = float(np.clip(5.0 - 4.0 * rank, 1.0, 5.0))
            scores[organ] = score
        pseudo_labels[fname] = scores

    logger.info(
        "Pseudo-labels computed for %d / %d training images.",
        len(pseudo_labels), len(train_filenames),
    )
    return pseudo_labels


# ---------------------------------------------------------------------------
# Method 2: beauty-axis projection pseudo-labels
# ---------------------------------------------------------------------------

def compute_all_pseudo_labels_beauty_axis(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    population_mean: np.ndarray,
    beauty_prototype: np.ndarray,
    population_mean_map: dict[str, np.ndarray] | None = None,
    beauty_prototype_map: dict[str, np.ndarray] | None = None,
    ethnicity_map: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Per-organ pseudo-labels via beauty-axis projection.

    Pipeline
    --------
    1. Build beauty_axis = beauty_prototype − population_mean.
    2. For each face / organ: scalar-project (face − population_mean) onto
       the organ-portion of beauty_axis.
    3. Percentile-rank projections across the dataset → score ∈ [1, 5].
       Higher projection (further along beauty axis) → higher score.
    """
    valid_fnames = [f for f in train_filenames if f in coords_cache]
    use_ethnicity = (
        population_mean_map is not None
        and beauty_prototype_map is not None
        and ethnicity_map is not None
    )

    global_axis = compute_beauty_axis(population_mean, beauty_prototype)
    axis_map: dict[str, np.ndarray] = {}
    if use_ethnicity:
        for eth in beauty_prototype_map:
            mu_eth = population_mean_map.get(eth, population_mean)
            axis_map[eth] = compute_beauty_axis(mu_eth, beauty_prototype_map[eth])
        logger.info(
            "Per-ethnicity beauty axes computed: %s",
            {eth: ax.shape for eth, ax in axis_map.items()},
        )

    organ_proj_all: dict[str, list[float]] = {o: [] for o in ORGAN_INDICES}

    for fname in tqdm(valid_fnames, desc="Pass 1 — projections", unit="face"):
        coords = coords_cache[fname]
        if use_ethnicity:
            eth  = ethnicity_map.get(fname, "Unknown")
            mu   = population_mean_map.get(eth, population_mean)
            axis = axis_map.get(eth, global_axis)
        else:
            mu, axis = population_mean, global_axis

        for organ, idxs in ORGAN_INDICES.items():
            proj = project_organ_onto_axis(coords, mu, axis, idxs)
            organ_proj_all[organ].append(proj)

    organ_sorted: dict[str, list[float]] = {
        o: sorted(v) for o, v in organ_proj_all.items()
    }
    logger.info(
        "Organ projection ranges (min→max): %s",
        {o: (round(v[0], 5), round(v[-1], 5)) for o, v in organ_sorted.items()},
    )

    pseudo_labels: dict[str, dict[str, float]] = {}

    for fname in tqdm(valid_fnames, desc="Pass 2 — pseudo scores", unit="face"):
        coords = coords_cache[fname]
        if use_ethnicity:
            eth  = ethnicity_map.get(fname, "Unknown")
            mu   = population_mean_map.get(eth, population_mean)
            axis = axis_map.get(eth, global_axis)
        else:
            mu, axis = population_mean, global_axis

        scores: dict[str, float] = {}
        for organ, idxs in ORGAN_INDICES.items():
            proj = project_organ_onto_axis(coords, mu, axis, idxs)
            sorted_vals = organ_sorted[organ]
            n = len(sorted_vals)
            rank = bisect.bisect_left(sorted_vals, proj) / n
            score = float(np.clip(1.0 + 4.0 * rank, 1.0, 5.0))
            scores[organ] = score
        pseudo_labels[fname] = scores

    logger.info(
        "Beauty-axis pseudo-labels computed for %d / %d training images.",
        len(pseudo_labels), len(train_filenames),
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
    Spearman ρ between mean organ pseudo-score and holistic rating.

    Prints a warning if ρ < 0.2 (pseudo-labels weakly aligned with GT).
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

    import pandas as pd

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

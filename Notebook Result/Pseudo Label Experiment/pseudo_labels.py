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


def compute_beauty_axis(
    population_mean: np.ndarray,
    beauty_prototype: np.ndarray,
) -> np.ndarray:
    """
    Beauty axis = direction from population mean → beauty prototype in face space.

    Following Valentine (1991) face space framework: faces are points in a
    multidimensional space. Said & Todorov (2011) showed attractiveness is
    better modelled as a *direction* in this space than as proximity to a
    fixed prototype. DeBruine & Jones (2007) demonstrate caricatured beautiful
    faces are mathematically farther from the mean but more attractive — they
    lie *along* the beauty axis, not near a single ideal point.

    Parameters
    ----------
    population_mean : np.ndarray, shape (468, 3)
        Element-wise mean of all training faces.
    beauty_prototype : np.ndarray, shape (468, 3)
        Mean of top-k% highest-rated training faces.

    Returns
    -------
    np.ndarray, shape (468, 3)
        Unnormalised beauty direction vector (per-landmark Δ).
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

    Positive  → organ deviates from the population mean *toward* the beauty
                direction (more attractive).
    Zero      → orthogonal to beauty direction (typical / neutral).
    Negative  → opposite of beauty direction (less attractive).

    Parameters
    ----------
    coords : np.ndarray, shape (468, 3)
    population_mean : np.ndarray, shape (468, 3)
    beauty_axis : np.ndarray, shape (468, 3)
    organ_indices : list[int]
    """
    deviation = coords[organ_indices] - population_mean[organ_indices]  # (n, 3)
    axis      = beauty_axis[organ_indices]                              # (n, 3)

    dev_flat  = deviation.ravel()
    axis_flat = axis.ravel()

    axis_norm = float(np.linalg.norm(axis_flat))
    if axis_norm < 1e-12:
        return 0.0
    return float(np.dot(dev_flat, axis_flat) / axis_norm)


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
# Beauty-axis pseudo-label computation
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
    Compute per-organ pseudo-labels via beauty-axis projection.

    Pipeline
    --------
    1. Build beauty_axis = beauty_prototype − population_mean.
    2. For each face / organ: scalar-project (face − population_mean) onto
       the organ-portion of beauty_axis.
    3. Percentile-rank projections across the dataset → score ∈ [1, 5].
       Direction: higher projection (further along the beauty axis) →
       higher score.

    Score interpretation
    --------------------
    Unlike the RMSE-from-prototype formulation, faces that geometrically
    differ from the mean *toward* the beauty direction receive high scores,
    even when they are far from the mean overall.  This addresses the
    DeBruine & Jones (2007) critique that attractive faces are not always
    average — they often lie along a directional axis in face space.

    Parameters
    ----------
    coords_cache : dict[str, np.ndarray]
        Filename → (468, 3) normalised coords.
    train_filenames : list[str]
        Ordered training-set filenames.
    population_mean : np.ndarray, shape (468, 3)
        Global population mean (fallback if ethnicity_map missing).
    beauty_prototype : np.ndarray, shape (468, 3)
        Global beauty prototype (top-k% mean).
    population_mean_map : dict[str, (468, 3)] | None
        Per-ethnicity population means.  H1 refinement.
    beauty_prototype_map : dict[str, (468, 3)] | None
        Per-ethnicity beauty prototypes.  H1 refinement.
    ethnicity_map : dict[str, str] | None
        Filename → ethnicity label.

    Returns
    -------
    dict[str, dict[str, float]]
        Filename → {organ_name: pseudo_score ∈ [1, 5]}.
    """
    valid_fnames = [f for f in train_filenames if f in coords_cache]
    use_ethnicity = (
        population_mean_map is not None
        and beauty_prototype_map is not None
        and ethnicity_map is not None
    )

    # Pre-compute beauty axes (per ethnicity or global)
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

    # ---- Pass 1: collect projections per organ ----
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

    # ---- Pass 2: percentile rank → score ∈ [1, 5] ----
    # Higher projection (further along beauty axis) → higher score.
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
            rank = bisect.bisect_left(sorted_vals, proj) / n  # ∈ [0, 1)
            # Higher projection → higher rank → higher score
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
# Subgroup generalisation of compute_ethnicity_avg_faces
# ---------------------------------------------------------------------------

def compute_subgroup_avg_faces(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    subgroup_map: dict[str, str],
    holistic_ratings: dict[str, float] | None = None,
    top_k_pct: float = 0.30,
) -> dict[str, np.ndarray]:
    """
    Generalisation of compute_ethnicity_avg_faces over arbitrary subgroup keys.

    Allows callers to build per-(ethnicity x gender) prototypes by passing a
    composite subgroup_map (e.g. "Asian_Female"), while keeping the original
    compute_ethnicity_avg_faces untouched.

    Returns dict mapping subgroup string -> (468, 3) prototype array.
    """
    groups_coords: dict[str, list[np.ndarray]] = {}
    groups_ratings: dict[str, list[float]] = {}

    for fname in train_filenames:
        if fname not in coords_cache:
            continue
        key = subgroup_map.get(fname, "Unknown")
        groups_coords.setdefault(key, []).append(coords_cache[fname])
        if holistic_ratings is not None:
            groups_ratings.setdefault(key, []).append(
                holistic_ratings.get(fname, 3.0)
            )

    result: dict[str, np.ndarray] = {}
    for key, coords_list in groups_coords.items():
        if holistic_ratings is not None and key in groups_ratings:
            result[key] = compute_beauty_prototype(
                coords_list, groups_ratings[key], top_k_pct=top_k_pct
            )
        else:
            result[key] = compute_universal_average_face(coords_list)
        logger.info(
            "Reference face for subgroup '%s' computed from %d faces.",
            key, len(coords_list),
        )
    return result


# ---------------------------------------------------------------------------
# Method: K-Means multi-prototype beauty axis (per ethnicity, K sub-prototypes)
# ---------------------------------------------------------------------------

def compute_all_pseudo_labels_beauty_axis_kmeans(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str],
    top_k_pct: float = 0.30,
    k_clusters: int = 3,
) -> dict[str, dict[str, float]]:
    """
    Multi-prototype variant: cluster the top-k% attractive faces of each
    ethnicity into K aesthetic subspaces, then assign each face to the
    closest beauty axis among the K candidates.

    Addresses feature cancellation when averaging diverse attractive faces
    (oval vs square vs heart) into a single prototype.

    Pipeline (per ethnicity):
        1. Take top-k% by holistic rating.
        2. KMeans(K) on flattened coords -> K sub-prototypes.
        3. K beauty axes = sub_prototype_k - population_mean_eth.
    Pipeline (per face):
        For each organ, project onto all K axes; keep the max projection
        (closest to that face's nearest beauty subspace). Then percentile-
        rank across the dataset -> score in [1, 5].
    """
    from sklearn.cluster import KMeans

    valid_fnames = [f for f in train_filenames if f in coords_cache]

    # ---- Per-ethnicity population means ----
    population_mean_map = compute_ethnicity_avg_faces(
        coords_cache, valid_fnames, ethnicity_map, holistic_ratings=None,
    )

    # ---- Per-ethnicity K sub-prototypes via KMeans ----
    sub_axes: dict[str, list[np.ndarray]] = {}
    for eth, mu_eth in population_mean_map.items():
        eth_fnames = [
            f for f in valid_fnames if ethnicity_map.get(f) == eth
        ]
        eth_coords = [coords_cache[f] for f in eth_fnames]
        eth_ratings = [holistic_ratings.get(f, 3.0) for f in eth_fnames]
        if not eth_coords:
            continue
        n_top = max(k_clusters, int(len(eth_ratings) * top_k_pct))
        top_idx = np.argsort(eth_ratings)[::-1][:n_top]
        top_coords = np.stack([eth_coords[i] for i in top_idx], axis=0)  # (N_top, 468, 3)
        flat = top_coords.reshape(len(top_idx), -1)                       # (N_top, 1404)

        km = KMeans(n_clusters=k_clusters, random_state=config.SEED, n_init=10)
        km.fit(flat)
        centroids = km.cluster_centers_.reshape(k_clusters, 468, 3).astype(np.float32)
        sub_axes[eth] = [
            (centroids[k] - mu_eth).astype(np.float32) for k in range(k_clusters)
        ]
        logger.info(
            "Ethnicity '%s': %d sub-prototypes from top %d / %d faces.",
            eth, k_clusters, n_top, len(eth_fnames),
        )

    # ---- Pass 1: collect max-projection per organ ----
    organ_proj_all: dict[str, list[float]] = {o: [] for o in ORGAN_INDICES}

    for fname in tqdm(valid_fnames, desc="K-means Pass 1 — projections", unit="face"):
        coords = coords_cache[fname]
        eth = ethnicity_map.get(fname, "Unknown")
        mu = population_mean_map.get(eth)
        axes = sub_axes.get(eth)
        if mu is None or axes is None:
            for organ in ORGAN_INDICES:
                organ_proj_all[organ].append(0.0)
            continue
        for organ, idxs in ORGAN_INDICES.items():
            proj_k = [
                project_organ_onto_axis(coords, mu, ax, idxs) for ax in axes
            ]
            organ_proj_all[organ].append(max(proj_k))

    organ_sorted: dict[str, list[float]] = {
        o: sorted(v) for o, v in organ_proj_all.items()
    }

    # ---- Pass 2: percentile-rank -> score ----
    pseudo_labels: dict[str, dict[str, float]] = {}
    for i, fname in enumerate(
        tqdm(valid_fnames, desc="K-means Pass 2 — scores", unit="face")
    ):
        scores: dict[str, float] = {}
        for organ in ORGAN_INDICES:
            proj = organ_proj_all[organ][i]
            sorted_vals = organ_sorted[organ]
            n = len(sorted_vals)
            rank = bisect.bisect_left(sorted_vals, proj) / n
            score = float(np.clip(1.0 + 4.0 * rank, 1.0, 5.0))
            scores[organ] = score
        pseudo_labels[fname] = scores

    logger.info(
        "K-means beauty-axis pseudo-labels computed for %d faces (K=%d).",
        len(pseudo_labels), k_clusters,
    )
    return pseudo_labels


# ---------------------------------------------------------------------------
# Method: Quantile remap of beauty-axis pseudo-labels to GT distribution shape
# ---------------------------------------------------------------------------

def compute_all_pseudo_labels_quantile(
    base_pseudo_labels: dict[str, dict[str, float]],
    holistic_ratings: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Reshape an existing pseudo-label set so its marginal distribution matches
    the empirical GT rating distribution.

    For each organ independently:
        rank face by current pseudo-score -> percentile in [0, 1)
        new_score = np.quantile(holistic_values, percentile)

    The rank ordering is preserved (Spearman rho identical to base) but the
    magnitude distribution morphs from uniform [1, 5] to a sample-matched
    Gaussian-ish shape. Intended to ease the L_reg vs L_rank conflict that
    arises when pseudo-labels saturate at the [1, 5] extremes.

    Parameters
    ----------
    base_pseudo_labels : dict[str, dict[str, float]]
        Output of any other compute_all_pseudo_labels_* function.
    holistic_ratings : dict[str, float]
        Source distribution to match. Typically the full training-set GT.
    """
    holistic_values = np.array(
        [v for v in holistic_ratings.values() if v is not None],
        dtype=np.float64,
    )
    if holistic_values.size == 0:
        raise ValueError("holistic_ratings is empty.")

    fnames = list(base_pseudo_labels.keys())
    organs = list(ORGAN_INDICES.keys())

    remapped: dict[str, dict[str, float]] = {f: {} for f in fnames}

    for organ in organs:
        values = np.array(
            [base_pseudo_labels[f].get(organ, 3.0) for f in fnames],
            dtype=np.float64,
        )
        order = np.argsort(values, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(values)) / max(len(values) - 1, 1)
        # Inverse CDF of empirical GT distribution
        new_scores = np.quantile(holistic_values, ranks)
        new_scores = np.clip(new_scores, 1.0, 5.0)
        for f, s in zip(fnames, new_scores):
            remapped[f][organ] = float(s)

    logger.info(
        "Quantile-matched pseudo-labels generated for %d faces over %d organs.",
        len(fnames), len(organs),
    )
    return remapped


# ---------------------------------------------------------------------------
# Pseudo-label-time MixUp: synthesise tail faces, then run beauty-axis
# projection on the augmented dataset so each synthetic face gets a proper
# geometric pseudo-label (rather than a linear interpolation of pseudo-
# scores from its two parents).
# ---------------------------------------------------------------------------

def _bucket_of_rating(rating: float, edges: tuple[float, ...]) -> int:
    """Map a rating to a discrete bucket index (0..len(edges))."""
    b = 0
    for e in edges:
        if rating >= e:
            b += 1
    return b


def generate_synthetic_tail_faces(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str],
    gender_map: dict[str, str] | None = None,
    n_synth_per_bucket: int = 500,
    bucket_edges: tuple[float, ...] = (2.0, 3.0, 4.0),
    target_buckets: tuple[int, ...] = (0, 3),  # Jelek + Cantik
    mixup_alpha: float = 0.4,
    seed: int = 42,
) -> tuple[
    dict[str, np.ndarray],  # aug_coords (original + synth)
    dict[str, float],       # aug_ratings
    dict[str, str],         # aug_ethnicity_map
    dict[str, str] | None,  # aug_gender_map
    list[str],              # aug_train_filenames (ordered)
]:
    """
    Synthesise additional faces for under-represented rating buckets by
    linearly mixing pairs of real faces within the same bucket.

    Each synthetic face inherits demographics from parent A. Its rating is
    α·r_A + (1-α)·r_B with α drawn from Beta(α, α). Filenames are stamped
    with the prefix ``synth_`` so downstream code can distinguish them.

    Parameters
    ----------
    n_synth_per_bucket : int
        How many synthetic faces to generate for each target bucket.
    target_buckets : tuple[int, ...]
        Bucket indices to augment. Default (0, 3) = Jelek + Cantik.
    mixup_alpha : float
        Beta distribution parameter. Smaller α → mixes closer to one parent
        (less extreme deviation); 0.4 mirrors training-time MixUp default.

    Returns
    -------
    aug_coords          : combined coords cache (original + synthetic)
    aug_ratings         : filename → rating (synthetic gets interpolated)
    aug_ethnicity_map   : filename → ethnicity (synthetic inherits parent A)
    aug_gender_map      : filename → gender (or None if not provided)
    aug_train_filenames : ordered list (originals first, then synthetic)
    """
    rng = np.random.RandomState(seed)

    # Group real faces by bucket
    bucket_to_real: dict[int, list[str]] = {}
    for fname in train_filenames:
        if fname not in coords_cache or fname not in holistic_ratings:
            continue
        b = _bucket_of_rating(holistic_ratings[fname], bucket_edges)
        bucket_to_real.setdefault(b, []).append(fname)

    aug_coords: dict[str, np.ndarray] = dict(coords_cache)
    aug_ratings: dict[str, float] = dict(holistic_ratings)
    aug_eth: dict[str, str] = dict(ethnicity_map)
    aug_gen: dict[str, str] | None = (
        dict(gender_map) if gender_map is not None else None
    )

    synth_filenames: list[str] = []

    for b in target_buckets:
        members = bucket_to_real.get(b, [])
        if len(members) < 2:
            logger.warning(
                "Bucket %d has %d members — skipping (need ≥2 to mix).",
                b, len(members),
            )
            continue

        for k in range(n_synth_per_bucket):
            # Sample two distinct parents from the same bucket
            i_a, i_b = rng.choice(len(members), size=2, replace=False)
            parent_a = members[i_a]
            parent_b = members[i_b]
            α = float(rng.beta(mixup_alpha, mixup_alpha))

            coords_mix = (
                α * coords_cache[parent_a] + (1.0 - α) * coords_cache[parent_b]
            ).astype(np.float32)
            rating_mix = (
                α * holistic_ratings[parent_a]
                + (1.0 - α) * holistic_ratings[parent_b]
            )

            synth_name = f"synth_b{b}_{k:05d}.jpg"
            aug_coords[synth_name] = coords_mix
            aug_ratings[synth_name] = float(rating_mix)
            aug_eth[synth_name] = ethnicity_map.get(parent_a, "Unknown")
            if aug_gen is not None:
                aug_gen[synth_name] = gender_map.get(parent_a, "Unknown")
            synth_filenames.append(synth_name)

        logger.info(
            "Generated %d synthetic faces for bucket %d (parents from %d real).",
            n_synth_per_bucket, b, len(members),
        )

    aug_train_filenames = list(train_filenames) + synth_filenames

    logger.info(
        "Augmented dataset: %d original + %d synthetic = %d faces.",
        len(train_filenames), len(synth_filenames), len(aug_train_filenames),
    )

    return aug_coords, aug_ratings, aug_eth, aug_gen, aug_train_filenames


def compute_all_pseudo_labels_axis_synthaug(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str],
    gender_map: dict[str, str] | None = None,
    n_synth_per_bucket: int = 500,
    bucket_edges: tuple[float, ...] = (2.0, 3.0, 4.0),
    target_buckets: tuple[int, ...] = (0, 3),
    mixup_alpha: float = 0.4,
    seed: int = 42,
    apply_quantile_remap: bool = True,
) -> tuple[
    dict[str, dict[str, float]],     # pseudo_labels (all faces)
    dict[str, np.ndarray],           # aug_coords
    dict[str, float],                # aug_ratings
    dict[str, str],                  # aug_ethnicity_map
    dict[str, str] | None,           # aug_gender_map
    list[str],                       # aug_train_filenames
]:
    """
    End-to-end pseudo-label-time MixUp pipeline:
      1. Synthesise tail faces (within-bucket).
      2. Recompute per-ethnicity population mean + beauty prototype on the
         augmented set (synthetic Cantik faces now contribute to top-30%).
      3. Run beauty-axis projection on every face (original + synthetic).
      4. (Optional) apply quantile remap to match GT distribution shape.

    Returns the pseudo-labels for ALL faces plus the augmented metadata
    needed to retrain the model with the expanded dataset.
    """
    # Step 1: synthesise
    aug_coords, aug_ratings, aug_eth, aug_gen, aug_fnames = (
        generate_synthetic_tail_faces(
            coords_cache=coords_cache,
            train_filenames=train_filenames,
            holistic_ratings=holistic_ratings,
            ethnicity_map=ethnicity_map,
            gender_map=gender_map,
            n_synth_per_bucket=n_synth_per_bucket,
            bucket_edges=bucket_edges,
            target_buckets=target_buckets,
            mixup_alpha=mixup_alpha,
            seed=seed,
        )
    )

    # Step 2: rebuild population mean + per-ethnicity beauty prototypes
    aug_coords_list = [aug_coords[f] for f in aug_fnames if f in aug_coords]
    aug_ratings_list = [aug_ratings[f] for f in aug_fnames if f in aug_coords]

    population_mean = compute_universal_average_face(aug_coords_list)
    beauty_prototype = compute_beauty_prototype(
        aug_coords_list, aug_ratings_list, top_k_pct=0.30,
    )

    pop_mean_eth_map = compute_ethnicity_avg_faces(
        aug_coords, aug_fnames, aug_eth, holistic_ratings=None,
    )
    beauty_proto_eth_map = compute_ethnicity_avg_faces(
        aug_coords, aug_fnames, aug_eth,
        holistic_ratings=aug_ratings, top_k_pct=0.30,
    )

    # Step 3: beauty-axis projection on entire augmented set
    pseudo_all = compute_all_pseudo_labels_beauty_axis(
        coords_cache=aug_coords,
        train_filenames=aug_fnames,
        population_mean=population_mean,
        beauty_prototype=beauty_prototype,
        population_mean_map=pop_mean_eth_map,
        beauty_prototype_map=beauty_proto_eth_map,
        ethnicity_map=aug_eth,
    )

    # Step 4: optional quantile remap (matches GT distribution shape)
    if apply_quantile_remap:
        pseudo_all = compute_all_pseudo_labels_quantile(
            base_pseudo_labels=pseudo_all,
            holistic_ratings=aug_ratings,
        )

    logger.info(
        "synthaug pseudo-labels: %d total (%d originals + %d synthetic).",
        len(pseudo_all), len(train_filenames),
        len(pseudo_all) - len(train_filenames),
    )

    return pseudo_all, aug_coords, aug_ratings, aug_eth, aug_gen, aug_fnames


# ---------------------------------------------------------------------------
# Symmetry & neoclassical canon pseudo-labels (structural alternative axis)
#
# These two methods use NON-prototype-based signals: a face's pseudo-score
# comes from how internally symmetric and proportion-correct it is, NOT
# from how close it sits to a beauty prototype. Provides a sanity check on
# whether structural priors (Schmid 2018 canons; bilateral symmetry, Said &
# Todorov 2011) add complementary signal on top of beauty-axis projection.
# ---------------------------------------------------------------------------

# Specific MediaPipe FaceMesh landmark indices used by canon ratios. These
# anchor points are stable across the 468-point topology and chosen to be
# semantically interpretable (no derivation from organ_indices needed).
_LM_FOREHEAD   = 10    # top of face oval
_LM_BROW_TOP   = 168   # top of nose bridge between eyebrows
_LM_NOSE_TIP   = 1
_LM_CHIN       = 152
_LM_LEYE_IN    = 133   # left eye inner corner (person's left)
_LM_LEYE_OUT   = 33    # left eye outer corner
_LM_REYE_IN    = 362   # right eye inner corner
_LM_REYE_OUT   = 263   # right eye outer corner
_LM_LMOUTH     = 61    # left mouth corner
_LM_RMOUTH     = 291   # right mouth corner
_LM_LNOSE      = 129   # left alar base
_LM_RNOSE      = 358   # right alar base


def compute_face_symmetry(coords: np.ndarray) -> dict[str, float]:
    """
    Per-organ bilateral symmetry score (lower = more symmetric).

    Reflects all landmarks across the face's vertical midline (x → -x in
    normalised coords) and pairs each organ's landmarks with the nearest
    reflected landmark from its anatomical mirror:
        left_eye  ↔ reflected(right_eye)
        right_eye ↔ reflected(left_eye)
        nose, mouth, jawline ↔ reflected(self)

    Returns
    -------
    dict[organ_name, float]
        Mean nearest-neighbour distance in reflected space.
    """
    reflected = coords.copy()
    reflected[:, 0] *= -1

    out: dict[str, float] = {}
    for organ, idxs in ORGAN_INDICES.items():
        org_coords = coords[idxs]                                 # (n, 3)
        if organ == "left_eye":
            mirror_src = reflected[ORGAN_INDICES["right_eye"]]
        elif organ == "right_eye":
            mirror_src = reflected[ORGAN_INDICES["left_eye"]]
        else:
            mirror_src = reflected[idxs]

        # Pairwise distance, take nearest mirror partner per landmark
        diff = org_coords[:, None, :] - mirror_src[None, :, :]    # (n, m, 3)
        dist = np.linalg.norm(diff, axis=-1)                      # (n, m)
        out[organ] = float(dist.min(axis=1).mean())
    return out


def compute_face_canons(coords: np.ndarray) -> dict[str, float]:
    """
    Per-organ neoclassical-canon deviation score (lower = closer to ideal).

    Six canon checks (each returns a non-negative relative error vs canon):
      1. Vertical thirds         — forehead↔brow ≈ brow↔nose ≈ nose↔chin
      2. Eye width consistency   — left_eye_w ≈ right_eye_w
      3. Eye separation canon    — inter-inner-corner ≈ eye width
      4. Mouth-eye width ratio   — mouth_width ≈ inter-outer-corner
      5. Nose-eye width ratio    — nose_w ≈ eye_w (Schmid 2018)
      6. Face golden ratio       — face_height / face_width ≈ 1.618

    Each canon error is then assigned to the organ(s) it geometrically
    involves; an organ's score is the sum of the canon errors that touch it.
    """
    def d(a: int, b: int) -> float:
        return float(np.linalg.norm(coords[a] - coords[b]))

    third_top   = d(_LM_FOREHEAD, _LM_BROW_TOP)
    third_mid   = d(_LM_BROW_TOP, _LM_NOSE_TIP)
    third_bot   = d(_LM_NOSE_TIP, _LM_CHIN)
    eye_w_l     = d(_LM_LEYE_OUT, _LM_LEYE_IN)
    eye_w_r     = d(_LM_REYE_OUT, _LM_REYE_IN)
    eye_w_avg   = 0.5 * (eye_w_l + eye_w_r)
    eye_sep     = d(_LM_LEYE_IN,  _LM_REYE_IN)
    mouth_w     = d(_LM_LMOUTH, _LM_RMOUTH)
    inter_outer = d(_LM_LEYE_OUT, _LM_REYE_OUT)
    nose_w      = d(_LM_LNOSE, _LM_RNOSE)
    face_h      = d(_LM_FOREHEAD, _LM_CHIN)
    # Approx face width via outermost jawline points (use eye-line outer corners as proxy)
    face_w      = inter_outer

    eps = 1e-8
    GOLDEN = 1.618

    thirds_mean = (third_top + third_mid + third_bot) / 3.0
    thirds_err  = (
        abs(third_top - thirds_mean)
        + abs(third_mid - thirds_mean)
        + abs(third_bot - thirds_mean)
    ) / (thirds_mean + eps)

    eyes_err     = abs(eye_w_l - eye_w_r) / (max(eye_w_l, eye_w_r) + eps)
    eye_sep_err  = abs(eye_sep - eye_w_avg) / (eye_w_avg + eps)
    mouth_eye_err = abs(mouth_w - inter_outer) / (inter_outer + eps)
    nose_eye_err  = abs(nose_w - eye_w_avg) / (eye_w_avg + eps)
    golden_err   = abs((face_h / (face_w + eps)) - GOLDEN) / GOLDEN

    # Distribute errors to organs they geometrically involve
    return {
        "left_eye":  eyes_err + mouth_eye_err + eye_sep_err,
        "right_eye": eyes_err + mouth_eye_err + eye_sep_err,
        "nose":      thirds_err + nose_eye_err,
        "mouth":     mouth_eye_err,
        "jawline":   thirds_err + golden_err,
    }


def _percentile_rank_to_scores(
    raw_scores_per_organ: dict[str, list[float]],
    fnames: list[str],
    invert: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Convert per-organ raw scalar scores → pseudo-labels in [1, 5] via
    percentile rank. Lower raw → higher score when invert=True (the default,
    since both symmetry and canon errors are "lower is better").
    """
    out: dict[str, dict[str, float]] = {f: {} for f in fnames}
    for organ, vals in raw_scores_per_organ.items():
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        for f, v in zip(fnames, vals):
            rank = bisect.bisect_left(sorted_vals, v) / n
            if invert:
                # Low error (rank near 0) → score near 5
                score = float(np.clip(5.0 - 4.0 * rank, 1.0, 5.0))
            else:
                score = float(np.clip(1.0 + 4.0 * rank, 1.0, 5.0))
            out[f][organ] = score
    return out


def compute_all_pseudo_labels_symmetry(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
) -> dict[str, dict[str, float]]:
    """Pseudo-labels from bilateral symmetry per organ (standalone)."""
    valid = [f for f in train_filenames if f in coords_cache]
    per_organ_raw: dict[str, list[float]] = {o: [] for o in ORGAN_INDICES}
    for f in tqdm(valid, desc="Symmetry — collecting", unit="face"):
        scores = compute_face_symmetry(coords_cache[f])
        for o, v in scores.items():
            per_organ_raw[o].append(v)
    logger.info(
        "Symmetry score ranges (min→max per organ): %s",
        {o: (round(min(v), 5), round(max(v), 5)) for o, v in per_organ_raw.items()},
    )
    return _percentile_rank_to_scores(per_organ_raw, valid, invert=True)


def compute_all_pseudo_labels_canons(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
) -> dict[str, dict[str, float]]:
    """Pseudo-labels from neoclassical-canon deviations (standalone)."""
    valid = [f for f in train_filenames if f in coords_cache]
    per_organ_raw: dict[str, list[float]] = {o: [] for o in ORGAN_INDICES}
    for f in tqdm(valid, desc="Canons — collecting", unit="face"):
        scores = compute_face_canons(coords_cache[f])
        for o, v in scores.items():
            per_organ_raw[o].append(v)
    logger.info(
        "Canon error ranges (min→max per organ): %s",
        {o: (round(min(v), 5), round(max(v), 5)) for o, v in per_organ_raw.items()},
    )
    return _percentile_rank_to_scores(per_organ_raw, valid, invert=True)


def compute_all_pseudo_labels_blend(
    components: list[tuple[dict[str, dict[str, float]], float]],
) -> dict[str, dict[str, float]]:
    """
    Weighted blend of multiple pseudo-label dicts.

    Output[face][organ] = Σ_i w_i * components[i][face][organ] / Σ_i w_i.

    Inputs are assumed to share the same set of (face, organ) keys; missing
    entries fall back to a neutral 3.0 so the blend is robust.

    Parameters
    ----------
    components : list[tuple[pseudo_dict, weight]]
        Each pseudo_dict maps filename → {organ_name: score in [1, 5]}.
        Weights need not sum to 1 (we normalise internally).
    """
    if not components:
        raise ValueError("Need at least one component to blend.")
    weight_sum = sum(w for _, w in components)
    if weight_sum <= 0:
        raise ValueError("Sum of blend weights must be positive.")

    # Union of all filenames across components
    all_fnames: set[str] = set()
    for d, _ in components:
        all_fnames.update(d.keys())
    organs = list(ORGAN_INDICES.keys())

    blended: dict[str, dict[str, float]] = {}
    for f in all_fnames:
        row: dict[str, float] = {}
        for o in organs:
            acc = 0.0
            for d, w in components:
                v = d.get(f, {}).get(o, 3.0)
                acc += w * v
            row[o] = float(np.clip(acc / weight_sum, 1.0, 5.0))
        blended[f] = row
    logger.info(
        "Blended pseudo-labels over %d components (weights sum=%.3f) for %d faces.",
        len(components), weight_sum, len(blended),
    )
    return blended


# ---------------------------------------------------------------------------
# Diagnostic report: per-subgroup + per-organ + per-bucket Spearman rho
# ---------------------------------------------------------------------------

def _spearman_safe(a: list[float], b: list[float]) -> tuple[float, int]:
    """Spearman rho with NaN guard for tiny / degenerate samples."""
    from scipy.stats import spearmanr

    if len(a) < 5:
        return float("nan"), len(a)
    rho, _ = spearmanr(a, b)
    if rho is None or np.isnan(rho):
        return float("nan"), len(a)
    return float(rho), len(a)


def pseudo_label_quality_report(
    pseudo_labels: dict[str, dict[str, float]],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str] | None = None,
    gender_map: dict[str, str] | None = None,
    method_name: str = "baseline",
    verbose: bool = True,
) -> dict:
    """
    Multi-axis pseudo-label quality audit.

    Reports Spearman rho between pseudo-score (mean over 5 organs) and GT
    holistic rating, broken down by:
        - global
        - per organ (5)
        - per ethnicity
        - per gender
        - per (ethnicity x gender)
        - per rating bucket (Jelek <2 / Avg 2-3 / Mid 3-4 / Cantik >=4)
        - Pearson r global (for reference)

    Returns a dict with all numbers; also prints a formatted block when
    verbose=True. Use method_name to tag rows for benchmark leaderboards.
    """
    from scipy.stats import pearsonr

    common = [f for f in pseudo_labels if f in holistic_ratings]
    if len(common) < 10:
        logger.warning("Too few samples (%d) for quality report.", len(common))
        return {"method": method_name, "n": len(common)}

    organs = list(ORGAN_INDICES.keys())

    mean_pseudo = {
        f: float(np.mean([pseudo_labels[f].get(o, 3.0) for o in organs]))
        for f in common
    }
    ratings = {f: holistic_ratings[f] for f in common}

    # ---- Global ----
    g_pseudo = [mean_pseudo[f] for f in common]
    g_rate = [ratings[f] for f in common]
    rho_global, _ = _spearman_safe(g_pseudo, g_rate)
    try:
        pearson_global, _ = pearsonr(g_pseudo, g_rate)
    except Exception:
        pearson_global = float("nan")

    # ---- Per organ ----
    per_organ: dict[str, tuple[float, int]] = {}
    for o in organs:
        a = [pseudo_labels[f].get(o, 3.0) for f in common]
        b = [ratings[f] for f in common]
        per_organ[o] = _spearman_safe(a, b)

    # ---- Per ethnicity ----
    per_eth: dict[str, tuple[float, int]] = {}
    if ethnicity_map is not None:
        eth_groups: dict[str, list[str]] = {}
        for f in common:
            eth = ethnicity_map.get(f, "Unknown")
            eth_groups.setdefault(eth, []).append(f)
        for eth, files in eth_groups.items():
            a = [mean_pseudo[f] for f in files]
            b = [ratings[f] for f in files]
            per_eth[eth] = _spearman_safe(a, b)

    # ---- Per gender ----
    per_gen: dict[str, tuple[float, int]] = {}
    if gender_map is not None:
        gen_groups: dict[str, list[str]] = {}
        for f in common:
            gen = gender_map.get(f, "Unknown")
            gen_groups.setdefault(gen, []).append(f)
        for gen, files in gen_groups.items():
            a = [mean_pseudo[f] for f in files]
            b = [ratings[f] for f in files]
            per_gen[gen] = _spearman_safe(a, b)

    # ---- Per (eth x gen) ----
    per_eth_gen: dict[str, tuple[float, int]] = {}
    if ethnicity_map is not None and gender_map is not None:
        cell_groups: dict[str, list[str]] = {}
        for f in common:
            key = f"{ethnicity_map.get(f, 'Unknown')}_{gender_map.get(f, 'Unknown')}"
            cell_groups.setdefault(key, []).append(f)
        for key, files in cell_groups.items():
            a = [mean_pseudo[f] for f in files]
            b = [ratings[f] for f in files]
            per_eth_gen[key] = _spearman_safe(a, b)

    # ---- Per rating bucket ----
    buckets = {
        "Jelek (<2)":   (1.0, 2.0),
        "Avg (2-3)":    (2.0, 3.0),
        "Mid (3-4)":    (3.0, 4.0),
        "Cantik (>=4)": (4.0, 5.01),
    }
    per_bucket: dict[str, tuple[float, int]] = {}
    for label, (lo, hi) in buckets.items():
        files = [f for f in common if lo <= ratings[f] < hi]
        a = [mean_pseudo[f] for f in files]
        b = [ratings[f] for f in files]
        per_bucket[label] = _spearman_safe(a, b)

    report = {
        "method": method_name,
        "n": len(common),
        "rho_global": rho_global,
        "pearson_global": float(pearson_global),
        "per_organ": per_organ,
        "per_ethnicity": per_eth,
        "per_gender": per_gen,
        "per_eth_gen": per_eth_gen,
        "per_bucket": per_bucket,
    }

    if verbose:
        print()
        print(f"========== Pseudo-Label Quality Report: {method_name} ==========")
        print(f"  n={len(common)}  Spearman rho={rho_global:.4f}  Pearson r={pearson_global:.4f}")
        print()
        print("  Per organ:")
        for o, (rho, n) in per_organ.items():
            print(f"    {o:<11} rho={rho:+.4f}  n={n}")
        if per_eth:
            print("  Per ethnicity:")
            for k, (rho, n) in per_eth.items():
                print(f"    {k:<11} rho={rho:+.4f}  n={n}")
        if per_gen:
            print("  Per gender:")
            for k, (rho, n) in per_gen.items():
                print(f"    {k:<11} rho={rho:+.4f}  n={n}")
        if per_eth_gen:
            print("  Per (eth x gen):")
            for k, (rho, n) in per_eth_gen.items():
                print(f"    {k:<20} rho={rho:+.4f}  n={n}")
        print("  Per rating bucket:")
        for k, (rho, n) in per_bucket.items():
            print(f"    {k:<14} rho={rho:+.4f}  n={n}")
        print("=" * (44 + len(method_name)))
        print()

    return report


# ---------------------------------------------------------------------------
# Benchmark leaderboard pretty-printer
# ---------------------------------------------------------------------------

def print_benchmark_leaderboard(results: dict[str, dict]) -> None:
    """
    Pretty-print a leaderboard ranking pseudo-label methods by global Spearman rho.

    Marks the top row with a star prefix. Columns: method, rho_global,
    pearson, n, plus per-subgroup highlights if present.
    """
    if not results:
        print("No benchmark results to display.")
        return

    rows = sorted(
        results.values(),
        key=lambda r: r.get("rho_global", float("-inf")),
        reverse=True,
    )

    print()
    print("=" * 110)
    print(f"{'rank':>4}  {'method':<28}  {'rho_glob':>9}  {'pearson':>8}  "
          f"{'AF':>7}  {'AM':>7}  {'CF':>7}  {'CM':>7}  "
          f"{'Jelek':>7}  {'Cantik':>7}  {'n':>5}")
    print("-" * 110)
    for i, r in enumerate(rows, start=1):
        method = r.get("method", "?")
        rho = r.get("rho_global", float("nan"))
        pear = r.get("pearson_global", float("nan"))
        n = r.get("n", 0)

        def _get(d: dict, key: str) -> float:
            v = d.get(key)
            return v[0] if isinstance(v, tuple) else float("nan")

        cells = r.get("per_eth_gen", {})
        af = _get(cells, "Asian_Female")
        am = _get(cells, "Asian_Male")
        cf = _get(cells, "Caucasian_Female")
        cm = _get(cells, "Caucasian_Male")
        buckets = r.get("per_bucket", {})
        rj = _get(buckets, "Jelek (<2)")
        rc = _get(buckets, "Cantik (>=4)")

        marker = "*" if i == 1 else " "
        print(f"{marker}{i:>3}  {method:<28}  {rho:>+9.4f}  {pear:>+8.4f}  "
              f"{af:>+7.3f}  {am:>+7.3f}  {cf:>+7.3f}  {cm:>+7.3f}  "
              f"{rj:>+7.3f}  {rc:>+7.3f}  {n:>5}")
    print("=" * 110)
    print()


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

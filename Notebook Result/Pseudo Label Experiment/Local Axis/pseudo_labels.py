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
# Method 3: kernel-weighted LOCAL beauty axis / LOCAL RMSE pseudo-labels
# ---------------------------------------------------------------------------
#
# Hypothesis A (Locality): a single global axis/prototype per ethnicity is
# too rigid across the whole face space. Computing a *local* reference around
# each face's rating should track the data manifold better, especially in the
# tails (very low / very high holistic ratings).
#
# Hypothesis B (Direction vs Proximity): a signed projection along an axis
# captures caricature-like extrapolation; isotropic RMSE cannot. We expose a
# `local_rmse` variant so the two effects can be disentangled via 2x2 ablation:
#
#                global ref          local ref
#     axis      axis_eth           local_axis    (existing)        (NEW)
#     RMSE      rmse_eth           local_rmse    (existing)        (NEW)
#
# Sanity: at very large bandwidth the kernel becomes ~uniform over the
# population and both NEW variants must converge to their global counterparts.


def compute_local_prototype(
    coords_list: list[np.ndarray],
    ratings: list[float],
    query_rating: float,
    bandwidth: float,
) -> np.ndarray:
    """
    Kernel-weighted mean face — a *local prototype* centred at ``query_rating``.

    Weights use a Gaussian kernel on the rating axis:
        w_i = exp( -(rating_i - query_rating)^2 / (2 * bandwidth^2) )

    Returns the weighted mean of the (468, 3) coordinate arrays. Used as a
    shared building block by both ``compute_local_beauty_axis`` and the
    ``local_rmse`` pseudo-label variant.

    Parameters
    ----------
    coords_list  : list of (468, 3) arrays.
    ratings      : list of float holistic ratings aligned with coords_list.
    query_rating : centre of the kernel on the rating axis.
    bandwidth    : Gaussian σ on the rating axis (e.g. 0.5).

    Notes
    -----
    For repeated calls over the same population, prefer the cached path used
    inside ``compute_all_pseudo_labels_local_*`` which pre-stacks coords once.
    """
    if not coords_list:
        raise ValueError("coords_list must be non-empty.")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be > 0.")

    coords_arr = np.stack(coords_list, axis=0).astype(np.float32)   # (N, 468, 3)
    ratings_arr = np.asarray(ratings, dtype=np.float32)             # (N,)

    # Log-domain Gaussian for numerical stability (subtract max before exp).
    log_w = -((ratings_arr - float(query_rating)) ** 2) / (2.0 * float(bandwidth) ** 2)
    log_w -= log_w.max()
    w = np.exp(log_w)                                               # (N,)
    w_sum = float(w.sum())
    if w_sum < 1e-12:
        # Should not happen after the log-stabilisation, but guard anyway.
        return coords_arr.mean(axis=0).astype(np.float32)

    prototype = np.einsum("i,ijk->jk", w, coords_arr) / w_sum
    return prototype.astype(np.float32)


def compute_local_beauty_axis(
    coords_list: list[np.ndarray],
    ratings: list[float],
    query_rating: float,
    bandwidth: float,
    delta: float = 0.5,
) -> np.ndarray:
    """
    Local beauty axis at ``query_rating``:

        axis = local_prototype(query_rating + delta)
             - local_prototype(query_rating - delta)

    Captures the *direction in face space* in which faces become more
    attractive in the neighbourhood of ``query_rating``. Unlike the global
    axis (top-30% prototype minus population mean), this axis is allowed to
    rotate as you move along the rating axis.
    """
    proto_hi = compute_local_prototype(coords_list, ratings, query_rating + delta, bandwidth)
    proto_lo = compute_local_prototype(coords_list, ratings, query_rating - delta, bandwidth)
    return (proto_hi - proto_lo).astype(np.float32)


# --- Internal: vectorised batch computation of local prototypes / axes -----
#
# A naive per-face call costs O(N) per face × N faces = O(N^2 * 468 * 3).
# For SCUT-FBP5500 (N≈5500) that's manageable but unnecessary: pre-stacking
# coords once and broadcasting weights over the population shaves >10×.


def _gaussian_weights(ratings_arr: np.ndarray, query: float, bandwidth: float) -> np.ndarray:
    log_w = -((ratings_arr - float(query)) ** 2) / (2.0 * float(bandwidth) ** 2)
    log_w -= log_w.max()
    return np.exp(log_w)


def _local_prototype_vec(
    coords_arr: np.ndarray,      # (N, 468, 3)
    ratings_arr: np.ndarray,     # (N,)
    query: float,
    bandwidth: float,
) -> np.ndarray:
    w = _gaussian_weights(ratings_arr, query, bandwidth)
    w_sum = float(w.sum())
    if w_sum < 1e-12:
        return coords_arr.mean(axis=0).astype(np.float32)
    return (np.einsum("i,ijk->jk", w, coords_arr) / w_sum).astype(np.float32)


def _percentile_rank_to_score(
    organ_values: dict[str, list[float]],
    higher_is_better: bool,
) -> dict[str, dict]:
    """
    Convert raw per-organ projections (or -RMSEs) into [1, 5] scores via
    percentile rank. Returns dict keyed by organ name, each entry holding
    {'sorted': sorted_list, 'min': float, 'max': float} for downstream lookup.
    """
    out: dict[str, dict] = {}
    for organ, vals in organ_values.items():
        sorted_vals = sorted(vals)
        out[organ] = {
            "sorted": sorted_vals,
            "min": float(sorted_vals[0]) if sorted_vals else 0.0,
            "max": float(sorted_vals[-1]) if sorted_vals else 0.0,
            "higher_is_better": higher_is_better,
        }
    return out


def _score_from_rank(value: float, organ_rank: dict) -> float:
    sorted_vals = organ_rank["sorted"]
    n = len(sorted_vals)
    if n == 0:
        return 3.0
    rank = bisect.bisect_left(sorted_vals, value) / n
    if organ_rank["higher_is_better"]:
        # higher value → higher score (mirror axis_eth)
        return float(np.clip(1.0 + 4.0 * rank, 1.0, 5.0))
    # lower value → higher score (mirror rmse_eth)
    return float(np.clip(5.0 - 4.0 * rank, 1.0, 5.0))


def _split_by_ethnicity(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str] | None,
) -> dict[str, dict]:
    """
    Group filenames by ethnicity and pre-stack coords + ratings per group.
    Faces without a rating or without a coords entry are skipped silently.
    If ethnicity_map is None, all faces are pooled under key '__all__'.
    """
    groups: dict[str, dict] = {}
    for fname in train_filenames:
        if fname not in coords_cache or fname not in holistic_ratings:
            continue
        eth = ethnicity_map.get(fname, "Unknown") if ethnicity_map else "__all__"
        g = groups.setdefault(eth, {"fnames": [], "coords": [], "ratings": []})
        g["fnames"].append(fname)
        g["coords"].append(coords_cache[fname])
        g["ratings"].append(float(holistic_ratings[fname]))

    for eth, g in groups.items():
        g["coords_arr"] = np.stack(g["coords"], axis=0).astype(np.float32)
        g["ratings_arr"] = np.asarray(g["ratings"], dtype=np.float32)
        logger.info(
            "Local-axis grouping — %s: %d faces (rating %.2f → %.2f).",
            eth, len(g["fnames"]),
            float(g["ratings_arr"].min()), float(g["ratings_arr"].max()),
        )
    return groups


def compute_all_pseudo_labels_local_axis(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str] | None = None,
    bandwidth: float = 0.5,
    delta: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Variant A — *local* + *directional* pseudo-labels.

    For each training face f with holistic rating r_f:
      1. Compute a kernel-weighted local prototype μ(r_f) (Gaussian on rating
         axis with given bandwidth) using only same-ethnicity faces when an
         ethnicity_map is provided (H1 refinement, mirrors axis_eth).
      2. Compute a local axis a(r_f) = μ(r_f + δ) - μ(r_f - δ).
      3. Per organ, project (f - μ(r_f)) onto the organ-portion of a(r_f).
      4. Percentile-rank projections across the dataset → score ∈ [1, 5]
         with HIGHER projection → HIGHER score (identical convention to
         ``compute_all_pseudo_labels_beauty_axis``).
    """
    groups = _split_by_ethnicity(coords_cache, train_filenames, holistic_ratings, ethnicity_map)
    organ_proj_all: dict[str, list[float]] = {o: [] for o in ORGAN_INDICES}
    fname_proj: dict[str, dict[str, float]] = {}

    for eth, g in groups.items():
        coords_arr = g["coords_arr"]
        ratings_arr = g["ratings_arr"]
        for fname, coords, r in tqdm(
            list(zip(g["fnames"], g["coords"], g["ratings"])),
            desc=f"local_axis — projections [{eth}, bw={bandwidth}]",
            unit="face",
        ):
            proto = _local_prototype_vec(coords_arr, ratings_arr, r, bandwidth)
            proto_hi = _local_prototype_vec(coords_arr, ratings_arr, r + delta, bandwidth)
            proto_lo = _local_prototype_vec(coords_arr, ratings_arr, r - delta, bandwidth)
            axis = (proto_hi - proto_lo).astype(np.float32)

            per_organ: dict[str, float] = {}
            for organ, idxs in ORGAN_INDICES.items():
                proj = project_organ_onto_axis(coords, proto, axis, idxs)
                per_organ[organ] = proj
                organ_proj_all[organ].append(proj)
            fname_proj[fname] = per_organ

    rank_table = _percentile_rank_to_score(organ_proj_all, higher_is_better=True)

    pseudo_labels: dict[str, dict[str, float]] = {}
    for fname, projs in fname_proj.items():
        pseudo_labels[fname] = {
            organ: _score_from_rank(projs[organ], rank_table[organ])
            for organ in ORGAN_INDICES
        }

    logger.info(
        "local_axis pseudo-labels computed for %d / %d faces (bw=%.3g, δ=%.3g).",
        len(pseudo_labels), len(train_filenames), bandwidth, delta,
    )
    return pseudo_labels


def compute_all_pseudo_labels_local_rmse(
    coords_cache: dict[str, np.ndarray],
    train_filenames: list[str],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str] | None = None,
    bandwidth: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Variant B — *local* + *isotropic* pseudo-labels (ablation).

    For each face f at holistic rating r_f:
      1. Compute kernel-weighted local prototype μ(r_f) (same as variant A).
      2. Per organ, RMSE between f and μ(r_f) — reuses ``compute_organ_mse``.
      3. Percentile-rank RMSEs across the dataset → score ∈ [1, 5] with
         LOWER RMSE → HIGHER score (identical convention to the global
         ``compute_all_pseudo_labels`` / rmse_eth path).

    Compared to ``local_axis``, this variant retains the locality benefit
    (Hypothesis A) but drops the directional projection (Hypothesis B).
    """
    groups = _split_by_ethnicity(coords_cache, train_filenames, holistic_ratings, ethnicity_map)
    organ_rmse_all: dict[str, list[float]] = {o: [] for o in ORGAN_INDICES}
    fname_rmse: dict[str, dict[str, float]] = {}

    for eth, g in groups.items():
        coords_arr = g["coords_arr"]
        ratings_arr = g["ratings_arr"]
        for fname, coords, r in tqdm(
            list(zip(g["fnames"], g["coords"], g["ratings"])),
            desc=f"local_rmse — distances [{eth}, bw={bandwidth}]",
            unit="face",
        ):
            proto = _local_prototype_vec(coords_arr, ratings_arr, r, bandwidth)

            per_organ: dict[str, float] = {}
            for organ, idxs in ORGAN_INDICES.items():
                rmse = compute_organ_mse(coords, proto, idxs)
                per_organ[organ] = rmse
                organ_rmse_all[organ].append(rmse)
            fname_rmse[fname] = per_organ

    rank_table = _percentile_rank_to_score(organ_rmse_all, higher_is_better=False)

    pseudo_labels: dict[str, dict[str, float]] = {}
    for fname, rmses in fname_rmse.items():
        pseudo_labels[fname] = {
            organ: _score_from_rank(rmses[organ], rank_table[organ])
            for organ in ORGAN_INDICES
        }

    logger.info(
        "local_rmse pseudo-labels computed for %d / %d faces (bw=%.3g).",
        len(pseudo_labels), len(train_filenames), bandwidth,
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

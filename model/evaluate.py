"""
evaluate.py — FaceRankNet
==========================
Evaluation metrics and local-score validity check.

Functions
---------
compute_pcc(preds, gts)                          — Pearson r
compute_mae(preds, gts)                          — Mean Absolute Error
compute_dpd(preds, gts, ethnicity_labels)        — Demographic Parity Difference
validate_local_scores(model, dataset, device)    — Spearman ρ check for all organs
run_full_evaluation(model, loader, device)       — Convenience wrapper
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

import config
from dataset import FaceDataset
from model import FaceRankNet
from organ_indices import ORGAN_INDICES

logger = logging.getLogger(__name__)

ORGAN_ORDER: list[str] = config.ORGAN_NAMES


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_pcc(preds: np.ndarray, gts: np.ndarray) -> float:
    """
    Pearson Correlation Coefficient between predictions and ground-truth.

    Parameters
    ----------
    preds, gts : np.ndarray — shape (N,), float.

    Returns
    -------
    float in [-1, 1].
    """
    if len(preds) < 2:
        return 0.0
    r, _ = pearsonr(preds.ravel(), gts.ravel())
    return float(r)


def compute_mae(preds: np.ndarray, gts: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    preds, gts : np.ndarray — shape (N,).

    Returns
    -------
    float ≥ 0.
    """
    return float(np.mean(np.abs(preds.ravel() - gts.ravel())))


def compute_dpd(
    preds: np.ndarray,
    gts: np.ndarray,
    ethnicity_labels: list[str | None],
) -> float:
    """
    Demographic Parity Difference = |MAE_Asian − MAE_Caucasian|.

    Images with unknown ethnicity are ignored.

    Parameters
    ----------
    preds            : np.ndarray, shape (N,)
    gts              : np.ndarray, shape (N,)
    ethnicity_labels : list of str or None, length N

    Returns
    -------
    float ≥ 0 (DPD).
    """
    p = preds.ravel()
    g = gts.ravel()
    eth = np.array(ethnicity_labels)

    asian_mask = eth == "Asian"
    cauc_mask = eth == "Caucasian"

    if asian_mask.sum() == 0 or cauc_mask.sum() == 0:
        logger.warning(
            "DPD: missing one ethnicity group (Asian=%d, Caucasian=%d). "
            "Returning 0.0.",
            asian_mask.sum(),
            cauc_mask.sum(),
        )
        return 0.0

    mae_asian = float(np.mean(np.abs(p[asian_mask] - g[asian_mask])))
    mae_cauc = float(np.mean(np.abs(p[cauc_mask] - g[cauc_mask])))
    return abs(mae_asian - mae_cauc)


# ---------------------------------------------------------------------------
# Local score validity (Spearman ρ check)
# ---------------------------------------------------------------------------

def validate_local_scores(
    model: FaceRankNet,
    dataset: FaceDataset,
    device: torch.device,
    max_samples: int = 500,
) -> bool:
    """
    Check that predicted local organ scores are positively correlated with
    pseudo-scores (Averageness Hypothesis validation).

    For each of the 5 organs, we compute the Spearman correlation between:
      - pseudo_score_i  (from the training pseudo-labels)
      - predicted local_score_i  (from the model)

    Returns True only if ALL 5 Spearman ρ values are > 0.

    Parameters
    ----------
    model      : FaceRankNet (eval mode, on ``device``).
    dataset    : FaceDataset with pseudo_labels populated.
    device     : torch device.
    max_samples: cap on number of samples to check (speed).

    Returns
    -------
    bool
    """
    model.eval()
    from dataset import collate_faces

    subset = torch.utils.data.Subset(
        dataset,
        list(range(min(max_samples, len(dataset)))),
    )
    loader = DataLoader(
        subset, batch_size=32, shuffle=False, collate_fn=collate_faces
    )

    all_pseudo: dict[str, list[float]] = {o: [] for o in ORGAN_ORDER}
    all_preds: dict[str, list[float]] = {o: [] for o in ORGAN_ORDER}

    with torch.no_grad():
        for batch in loader:
            subgraphs = {
                k: v.to(device) for k, v in batch["subgraphs"].items()
            }
            out = model(subgraphs)
            local_scores = out["local_scores"]

            pseudo_batch = batch["pseudo_scores"].numpy()  # (B, 5)

            for o_idx, organ in enumerate(ORGAN_ORDER):
                score_vals = local_scores[organ].cpu().numpy()
                for s in score_vals:
                    all_preds[organ].append(float(s))
                for ps in pseudo_batch[:, o_idx]:
                    all_pseudo[organ].append(float(ps))

    all_positive = True
    for organ in ORGAN_ORDER:
        rho, pval = spearmanr(all_pseudo[organ], all_preds[organ])
        logger.info(
            "  [%s] Spearman ρ = %.4f  (p=%.4f)", organ, rho, pval
        )
        if rho <= 0:
            all_positive = False

    return all_positive


# ---------------------------------------------------------------------------
# Full evaluation convenience wrapper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_full_evaluation(
    model: FaceRankNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Run inference over a DataLoader and return PCC, MAE, and DPD.

    Parameters
    ----------
    model  : FaceRankNet in eval mode.
    loader : DataLoader yielding collated face batches.
    device : torch device.

    Returns
    -------
    dict with keys: 'pcc', 'mae', 'dpd'.
    """
    model.eval()

    all_preds: list[float] = []
    all_gts: list[float] = []
    all_eth: list[str | None] = []

    for batch in loader:
        subgraphs = {k: v.to(device) for k, v in batch["subgraphs"].items()}
        out = model(subgraphs)
        preds = out["global_score"].cpu().numpy()
        gts = batch["ratings"].numpy()
        eths = batch["ethnicities"]

        all_preds.extend(preds.tolist())
        all_gts.extend(gts.tolist())
        all_eth.extend(eths)

    preds_arr = np.array(all_preds, dtype=np.float32)
    gts_arr = np.array(all_gts, dtype=np.float32)

    pcc = compute_pcc(preds_arr, gts_arr)
    mae = compute_mae(preds_arr, gts_arr)
    dpd = compute_dpd(preds_arr, gts_arr, all_eth)

    return {"pcc": pcc, "mae": mae, "dpd": dpd}

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
compute_fusion_sensitivity(model, loader, device) — Gradient-based sensitivity diagnostic
print_fusion_sensitivity(report)                 — Pretty-print sensitivity report
inspect_fusion_weights(model)                    — Softmax fusion weights per organ
evaluate_organ_contributions(model, loader, dev) — Per-organ stats table
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
# Fusion sensitivity diagnostic
# ---------------------------------------------------------------------------

def compute_fusion_sensitivity(
    model: FaceRankNet,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """
    Verify that local_scores actually shape global_score in the trained model.

    For each batch we compute:
      - ``|∂global_score / ∂local_score_i|`` averaged across samples, per organ.
      - ``|∂global_score / ∂global_embed|`` averaged across samples + embed dims.

    Interpretation
    --------------
    The fair aggregate comparison is between L1 norms of the full gradient
    vector with respect to each input group:

        Σ_local = Σ_i  mean_batch |∂g/∂local_i|        (5 organs)
        Σ_embed = mean_batch ||∂g/∂global_embed||_1
                = mean_batch_dim × embed_dim           (256 dims)

    If Σ_local / Σ_embed << 1 the head is effectively ignoring local_scores.

    Returns
    -------
    dict with keys:
        per_organ                : {organ: mean |∂g/∂local_i|}
        global_embed_per_dim     : mean |∂g/∂global_embed[k]| over embed dims
        global_embed_l1_per_face : mean ||∂g/∂global_embed||_1 per face
        embed_dim                : number of embed dimensions
        ratio_local_to_embed_pd  : per-dim ratio (diagnostic only)
        sigma_local              : Σ over 5 organs of per_organ[o]
        sigma_embed              : same as global_embed_l1_per_face
        ratio_sigma              : sigma_local / sigma_embed (verdict)
        fusion_mode              : the architecture's setting
        n_batches                : how many batches we averaged
    """
    model.eval()
    sens_local: dict[str, list[float]] = {o: [] for o in ORGAN_ORDER}
    sens_embed: list[float] = []
    n_batches = 0

    for batch in loader:
        if n_batches >= max_batches:
            break
        subgraphs = {k: v.to(device) for k, v in batch["subgraphs"].items()}

        with torch.enable_grad():
            out = model(subgraphs)
            global_score = out["global_score"]
            local_scores = out["local_scores"]
            global_embed = out.get("global_embed")

            for organ in ORGAN_ORDER:
                ls = local_scores[organ]
                grads = torch.autograd.grad(
                    global_score.sum(), ls,
                    retain_graph=True, create_graph=False, allow_unused=True,
                )[0]
                if grads is None:
                    sens_local[organ].extend([0.0] * global_score.shape[0])
                else:
                    sens_local[organ].extend(grads.abs().detach().cpu().tolist())

            if global_embed is not None and global_embed.requires_grad:
                grads = torch.autograd.grad(
                    global_score.sum(), global_embed,
                    retain_graph=False, create_graph=False, allow_unused=True,
                )[0]
                if grads is not None:
                    sens_embed.extend(
                        grads.abs().mean(dim=-1).detach().cpu().tolist()
                    )

        n_batches += 1

    per_organ_mean = {
        o: float(np.mean(s)) if s else 0.0 for o, s in sens_local.items()
    }
    embed_per_dim = float(np.mean(sens_embed)) if sens_embed else 0.0
    embed_dim = config.GAT_HIDDEN_DIM * config.GAT_NUM_HEADS
    embed_l1_per_face = embed_per_dim * embed_dim

    if embed_per_dim > 1e-12:
        ratio_pd = {o: v / embed_per_dim for o, v in per_organ_mean.items()}
    else:
        ratio_pd = {o: float("inf") if v > 0 else 0.0 for o, v in per_organ_mean.items()}

    sigma_local = float(sum(per_organ_mean.values()))
    sigma_embed = embed_l1_per_face
    if sigma_embed > 1e-12:
        ratio_sigma = sigma_local / sigma_embed
    else:
        ratio_sigma = float("inf") if sigma_local > 0 else 0.0

    return {
        "per_organ":               per_organ_mean,
        "global_embed_per_dim":    embed_per_dim,
        "global_embed_l1_per_face": embed_l1_per_face,
        "embed_dim":               embed_dim,
        "ratio_local_to_embed_pd": ratio_pd,
        "sigma_local":             sigma_local,
        "sigma_embed":             sigma_embed,
        "ratio_sigma":             ratio_sigma,
        "fusion_mode":             getattr(model, "fusion_mode", "?"),
        "n_batches":               n_batches,
    }


def print_fusion_sensitivity(report: dict) -> None:
    """Pretty-print the sensitivity report from compute_fusion_sensitivity."""
    mode = report.get("fusion_mode", "?")
    embed_pd = report["global_embed_per_dim"]
    embed_l1 = report["global_embed_l1_per_face"]
    embed_dim = report["embed_dim"]

    print("=" * 68)
    print(f"  Fusion sensitivity  (mode={mode}, n_batches={report['n_batches']})")
    print("-" * 68)
    print("  Per-dim diagnostic (each input scalar):")
    print(f"    mean |∂g / ∂global_embed[k]|  = {embed_pd:.6f}   ({embed_dim} dims)")
    print("-" * 68)
    print(f"  {'organ':<11} | {'mean |∂g/∂local_i|':>22} | {'per-dim ratio':>15}")
    print("-" * 68)
    for organ in ORGAN_ORDER:
        v = report["per_organ"][organ]
        r = report["ratio_local_to_embed_pd"][organ]
        marker = "  <- dominant" if r > 1.0 else ""
        print(f"  {organ:<11} | {v:>22.6f} | {r:>15.3f}{marker}")
    print("=" * 68)
    sigma_local = report["sigma_local"]
    sigma_embed = report["sigma_embed"]
    ratio_sigma = report["ratio_sigma"]
    print("  Aggregate L1-norm of gradient (the verdict):")
    print(f"    Σ_local = Σ over 5 organs       = {sigma_local:.6f}")
    print(f"    Σ_embed = mean × {embed_dim} dims         = {sigma_embed:.6f}")
    print(f"    ratio (Σ_local / Σ_embed)       = {ratio_sigma:.4f}")
    print("=" * 68)
    if mode == "score_aware":
        if ratio_sigma < 0.05:
            print(f"  ✗ FAIL — Σ_local is < 5% of Σ_embed.")
            print(f"    local_scores carry negligible aggregate influence on global_score.")
        elif ratio_sigma < 0.20:
            print(f"  ~ MARGINAL — Σ_local is {ratio_sigma*100:.1f}% of Σ_embed.")
        else:
            print(f"  ✓ PASS — Σ_local / Σ_embed = {ratio_sigma:.2f}.")
    elif mode == "fusion_weight":
        print("  (fusion_weight mode: ratios equal softmax(fusion_weights);")
        print("   Σ_embed = 0 because global_embed is not in the forward path.)")
    print()


# ---------------------------------------------------------------------------
# Fusion weight inspection
# ---------------------------------------------------------------------------

def inspect_fusion_weights(model: FaceRankNet) -> dict[str, float]:
    """
    Extract per-organ fusion weights (softmax of learnable parameters).

    Parameters
    ----------
    model : FaceRankNet — may be in train or eval mode.

    Returns
    -------
    dict[organ_name → weight], ordered by descending weight.
    """
    with torch.no_grad():
        weights = torch.softmax(model.fusion_weights, dim=0).cpu().numpy()

    result = {organ: float(w) for organ, w in zip(ORGAN_ORDER, weights)}
    ranked = sorted(result.items(), key=lambda x: x[1], reverse=True)

    logger.info("--- Per-organ fusion weights (softmax) ---")
    for rank, (organ, w) in enumerate(ranked, 1):
        bar = "#" * int(w * 40)
        logger.info("  %d. %-14s  %.4f  %s", rank, organ, w, bar)

    return dict(ranked)


@torch.no_grad()
def evaluate_organ_contributions(
    model: FaceRankNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Per-organ statistics: fusion weight, mean/std local score, and weighted
    contribution to the global score.

    Parameters
    ----------
    model  : FaceRankNet in eval mode.
    loader : DataLoader yielding collated face batches.
    device : torch device.

    Returns
    -------
    dict[organ_name → dict] where inner dict has keys:
        'fusion_weight'       — softmax weight (fixed across samples)
        'mean_local_score'    — mean predicted local score over the dataset
        'std_local_score'     — std of predicted local score
        'weighted_contrib'    — fusion_weight × mean_local_score
    """
    model.eval()

    all_local: dict[str, list[float]] = {o: [] for o in ORGAN_ORDER}

    for batch in loader:
        subgraphs = {k: v.to(device) for k, v in batch["subgraphs"].items()}
        out = model(subgraphs)
        for organ in ORGAN_ORDER:
            scores = out["local_scores"][organ].cpu().numpy().tolist()
            all_local[organ].extend(scores)

    with torch.no_grad():
        weights = torch.softmax(model.fusion_weights, dim=0).cpu().numpy()

    results: dict[str, dict[str, float]] = {}
    logger.info("--- Per-organ contribution analysis ---")
    logger.info(
        "  %-14s  %7s  %10s  %9s  %12s",
        "Organ", "Weight", "Mean Score", "Std Score", "Contrib",
    )
    for organ, w in zip(ORGAN_ORDER, weights):
        arr = np.array(all_local[organ], dtype=np.float32)
        mean_s = float(arr.mean())
        std_s = float(arr.std())
        contrib = float(w) * mean_s
        results[organ] = {
            "fusion_weight": float(w),
            "mean_local_score": mean_s,
            "std_local_score": std_s,
            "weighted_contrib": contrib,
        }
        logger.info(
            "  %-14s  %7.4f  %10.4f  %9.4f  %12.4f",
            organ, float(w), mean_s, std_s, contrib,
        )

    return results


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

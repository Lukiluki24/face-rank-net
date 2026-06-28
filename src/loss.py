"""
loss.py — FaceRankNet
======================
Three loss components for FaceRankNet training.

Functions
---------
l_reg(global_pred, global_gt)          — MSE regression loss
l_rank(scores_A, scores_B, organ_mask) — Pairwise ranking loss (per organ)
l_div(local_scores)                    — Diversity regularisation (-Var + boundary)

Combined in train.py as:
    total_loss = l_reg + LRANK_WEIGHT * l_rank + LDIV_WEIGHT * l_div
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import config

ORGAN_ORDER: list[str] = config.ORGAN_NAMES


# ---------------------------------------------------------------------------
# L_reg — MSE regression against holistic ground-truth
# ---------------------------------------------------------------------------

def l_reg(
    global_pred: torch.Tensor,
    global_gt: torch.Tensor,
) -> torch.Tensor:
    """Mean Squared Error between predicted and true beauty scores."""
    return F.mse_loss(global_pred, global_gt.to(global_pred.dtype))


# ---------------------------------------------------------------------------
# L_rank — Pairwise ranking loss
# ---------------------------------------------------------------------------

def l_rank(
    scores_A: dict[str, torch.Tensor],
    scores_B: dict[str, torch.Tensor],
    organ_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise ranking loss summed over organs.

    For each organ where pseudo_score(A) > pseudo_score(B) (organ_mask=True),
    penalise if the model's local score for A is not greater than B:

        L_rank = Σ_organ  log(1 + exp(score_B[organ] - score_A[organ]))

    Parameters
    ----------
    scores_A, scores_B : dict[str, Tensor] — shape (B,) per organ.
    organ_mask         : Tensor, shape (B, 5) — bool.

    Returns
    -------
    Tensor — scalar ranking loss.
    """
    total = torch.tensor(0.0, device=organ_mask.device)
    count = 0

    for o_idx, organ in enumerate(ORGAN_ORDER):
        s_a = scores_A[organ]
        s_b = scores_B[organ]
        mask = organ_mask[:, o_idx].float()
        pair_loss = torch.log1p(torch.exp(s_b - s_a))
        total = total + (mask * pair_loss).sum()
        count += mask.sum()

    if count > 0:
        return total / count
    return total


# ---------------------------------------------------------------------------
# L_div — Diversity regularisation
# ---------------------------------------------------------------------------

def l_div(local_scores: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Diversity regularisation: within-organ variance across the batch + boundary penalty.

    1. L_within = -mean_organs( Var_batch(score_organ) )
       Encourages each organ to discriminate between faces.

    2. L_boundary = mean( relu(1.2 - s) + relu(s - 4.8) )
       Penalises scores that saturate near the (1, 5) boundary.

    Returns
    -------
    Tensor — scalar (minimising = more diverse, less saturated).
    """
    stacked = torch.stack(
        [local_scores[o] for o in ORGAN_ORDER], dim=-1
    )  # (B, 5)

    within_var = stacked.var(dim=0, correction=0).mean()
    l_within = -within_var

    l_boundary = (
        torch.relu(1.2 - stacked) + torch.relu(stacked - 4.8)
    ).mean()

    return l_within + l_boundary

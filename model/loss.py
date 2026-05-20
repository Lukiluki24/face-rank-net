"""
loss.py — FaceRankNet
======================
Implements the three loss components and GradNorm.

Functions
---------
l_reg(global_pred, global_gt)           — MSE regression loss
l_rank(scores_A, scores_B, organ_mask) — Pairwise ranking loss (per organ)
l_div(local_scores)                    — Diversity regularisation (-Var)

Class
-----
GradNorm
    Dynamic loss-weight balancer following Chen et al. 2018 (NeurIPS).
    Tracks L0 (initial losses) and updates λ_i after each backward pass.
    Weights are kept non-negative via torch.clamp.

All lambdas start at 1.0 and are stored as nn.Parameter so the optimiser
does NOT touch them — GradNorm updates them manually via .data assignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
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
    """
    Standard Mean Squared Error between predicted and true beauty scores.

    Parameters
    ----------
    global_pred : Tensor, shape (B,)
    global_gt   : Tensor, shape (B,)

    Returns
    -------
    Tensor — scalar MSE.
    """
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
    scores_A : dict[str, Tensor]
        Local scores for face A — shape (B,) per organ.
    scores_B : dict[str, Tensor]
        Local scores for face B — shape (B,) per organ.
    organ_mask : Tensor, shape (B, 5) — bool
        True where pseudo_score_A > pseudo_score_B for that organ.

    Returns
    -------
    Tensor — scalar ranking loss.
    """
    total = torch.tensor(0.0, device=organ_mask.device)
    count = 0

    for o_idx, organ in enumerate(ORGAN_ORDER):
        s_a = scores_A[organ]                   # (B,)
        s_b = scores_B[organ]                   # (B,)
        mask = organ_mask[:, o_idx].float()     # (B,)

        # Soft max-margin loss: log(1 + exp(s_B - s_A)) weighted by mask
        pair_loss = torch.log1p(torch.exp(s_b - s_a))  # (B,)
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
    Diversity regularisation: negative variance of organ scores.

    Encourages the model to produce diverse local scores rather than
    collapsing all organs to the same value.

        L_div = -Var( [s_left_eye, s_right_eye, s_nose, s_mouth, s_jawline] )

    Parameters
    ----------
    local_scores : dict[str, Tensor]
        Each value shape (B,).

    Returns
    -------
    Tensor — scalar (negative variance, so minimising → maximising diversity).
    """
    stacked = torch.stack(
        [local_scores[o] for o in ORGAN_ORDER], dim=-1
    )  # (B, 5)
    var = stacked.var(dim=-1).mean()   # mean variance over batch
    return -var


# ---------------------------------------------------------------------------
# GradNorm — Dynamic loss weighting (Chen et al. 2018)
# ---------------------------------------------------------------------------

class GradNorm:
    """
    GradNorm adaptive loss weighting (Chen et al., NeurIPS 2018).

    Maintains a set of non-negative loss weights λ_i, one per task.
    After each forward pass, call ``update()`` to rebalance weights based
    on the gradient norms and loss ratios relative to their initial values.

    Parameters
    ----------
    model : nn.Module
        The shared network (FaceRankNet).  The last layer's parameters
        are used as the reference shared parameters for gradient norms.
    num_tasks : int
        Number of loss terms (default 3: L_reg, L_rank, L_div).
    alpha : float
        GradNorm restoring force hyper-parameter (default 1.5).
    """

    def __init__(
        self,
        model: nn.Module,
        num_tasks: int = config.NUM_TASKS,
        alpha: float = config.GRADNORM_ALPHA,
    ) -> None:
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Learnable loss weights (NOT optimised by the task optimiser)
        self.lambdas: torch.Tensor = torch.ones(
            num_tasks, dtype=torch.float32, requires_grad=False
        )

        # Initial losses L0 — set on first call to update()
        self.L0: torch.Tensor | None = None

        # Reference shared parameters: use fusion_weights as proxy
        # (last shared layer before task-specific heads)
        self._shared_params: list[nn.Parameter] = list(
            model.fusion_weights.unsqueeze(0)  # wrap so it looks like a param list
        ) if False else self._get_shared_params(model)

    @staticmethod
    def _get_shared_params(model: nn.Module) -> list[nn.Parameter]:
        """Return params that receive gradients from all tasks.

        The final Linear(32→1) of every OrganGAT MLP is the last computation
        before local_scores, so both L_reg (via global_score) and L_rank (via
        local_scores) produce non-zero gradients here — unlike input_proj which
        only belongs to one organ and is only reached by L_reg.
        """
        params: list[nn.Parameter] = []
        for organ in config.ORGAN_NAMES:
            params.extend(model.organ_gats[organ].mlp[-1].parameters())
        return params

    def update(
        self,
        losses: list[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Compute the GradNorm loss, update lambdas, and return the
        weighted total task loss for the main backward pass.

        Steps (Chen et al. 2018, Algorithm 1):
        1. Compute gradient norms ||∇_W (λ_i · L_i)|| for each task.
        2. Compute loss ratios  r_i = L_i / L0_i.
        3. Compute mean gradient norm  ḡ.
        4. Compute target gradient norm  G_i = ḡ · (r_i / r̄)^α.
        5. GradNorm loss = Σ_i |G_i(t) - ||∇_W(λ_i L_i)||_2|.
        6. Back-prop GradNorm loss through lambdas, update them, clamp ≥ 0.
        7. Re-normalise: λ_i ← λ_i × num_tasks / Σ_j λ_j.

        Parameters
        ----------
        losses : list[Tensor]
            Task losses [L_reg, L_rank, L_div] — detached scalars are fine
            because we re-weight them here.
        optimizer : torch.optim.Optimizer
            Main task optimiser (used to zero grad on lambda params).

        Returns
        -------
        Tensor
            Weighted total loss = Σ_i λ_i · L_i  (for main backward).
        """
        assert len(losses) == self.num_tasks, (
            f"Expected {self.num_tasks} losses, got {len(losses)}"
        )

        device = losses[0].device

        # Move lambdas to device on first call
        if self.lambdas.device != device:
            self.lambdas = self.lambdas.to(device)

        # Initialise L0 on first call
        if self.L0 is None:
            self.L0 = torch.stack(
                [l.detach().clone() for l in losses]
            ).to(device)
            self.L0 = torch.clamp(self.L0, min=1e-8)

        # Enable grad for lambdas temporarily
        lambdas_var = self.lambdas.clone().requires_grad_(True)

        # ---- Gradient norms G_i ----
        grad_norms: list[torch.Tensor] = []
        for i, loss_i in enumerate(losses):
            weighted = lambdas_var[i] * loss_i
            grads = torch.autograd.grad(
                weighted,
                self._shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            # Filter None grads
            valid = [g for g in grads if g is not None]
            if valid:
                norm = torch.stack([g.norm() for g in valid]).mean()
            else:
                norm = torch.tensor(0.0, device=device, requires_grad=True)
            grad_norms.append(norm)

        g_norms = torch.stack(grad_norms)  # (T,)

        # ---- Loss ratios ----
        l_current = torch.stack(
            [l.detach() for l in losses]
        ).to(device).clamp(min=1e-8)
        loss_ratio = l_current / self.L0           # (T,)  r_i
        mean_ratio = loss_ratio.mean()             # r̄

        # ---- Target norms ----
        g_bar = g_norms.detach().mean()            # ḡ
        target = (g_bar * (loss_ratio / mean_ratio) ** self.alpha).detach()

        # ---- GradNorm loss ----
        gradnorm_loss = torch.abs(g_norms - target).sum()

        # ---- Update lambdas ----
        # retain_graph=True keeps the graph alive for total_loss.backward()
        # which train.py calls immediately after this function returns.
        gradnorm_loss.backward(retain_graph=True)
        with torch.no_grad():
            if lambdas_var.grad is not None:
                grad_step = lambdas_var.grad.detach()
                self.lambdas = self.lambdas - 1e-3 * grad_step

        # Clamp and re-normalise
        self.lambdas = torch.clamp(self.lambdas, min=0.0)
        lam_sum = self.lambdas.sum()
        if lam_sum > 1e-8:
            self.lambdas = (
                self.lambdas * self.num_tasks / lam_sum
            ).detach()

        # ---- Weighted total loss ----
        total_loss = sum(
            self.lambdas[i].item() * losses[i]
            for i in range(self.num_tasks)
        )
        return total_loss

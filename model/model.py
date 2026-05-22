"""
model.py — FaceRankNet
=======================
Two nn.Module classes:

  OrganGAT
      Independent Graph Attention Network for one anatomical sub-graph.
      Architecture:
          Linear(3 → hidden_dim)
          → GATConv(hidden_dim, hidden_dim, num_heads)
          → mean-attention pooling over nodes
          → MLP(hidden_dim × num_heads → 32 → 1)
          → 4 × sigmoid(x) + 1  (enforces score ∈ (1, 5))

  FaceRankNet
      One OrganGAT per organ (5 total).
      Cross-organ MultiheadAttention fuses organ embeddings → global score.
      Learnable fusion_weights kept for interpretability only.
      forward() returns a dict with keys:
          'local_scores'  : dict[str, Tensor]  – one scalar per organ
          'global_score'  : Tensor             – cross-organ attended score, shape (B,)
          'organ_weights' : Tensor             – softmax(w), shape (5,) [interpretability]
          'attn_weights'  : Tensor             – cross-organ attention map, shape (B, 5, 5)

Reproducibility: torch.manual_seed(42), dgl.seed(42) set at module level.
No pixel data is loaded anywhere in this file.
"""

from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

torch.manual_seed(config.SEED)
dgl.seed(config.SEED)

ORGAN_ORDER: list[str] = config.ORGAN_NAMES


# ---------------------------------------------------------------------------
# Score range enforcement
# ---------------------------------------------------------------------------

def scale_to_score(x: torch.Tensor) -> torch.Tensor:
    """Map any real-valued tensor to (1, 5) via 4 × sigmoid(x) + 1."""
    return 4.0 * torch.sigmoid(x) + 1.0


# ---------------------------------------------------------------------------
# OrganGAT
# ---------------------------------------------------------------------------

class OrganGAT(nn.Module):
    """
    Graph Attention Network for a single anatomical organ sub-graph.

    Parameters
    ----------
    in_feats : int
        Input node feature dimension (3 for (x, y, z)).
    hidden_dim : int
        Hidden dimension before/after GAT.
    num_heads : int
        Number of attention heads in GATConv.
    dropout : float
        Dropout rate applied after the input projection and after GAT.
    """

    def __init__(
        self,
        in_feats: int = config.NODE_FEAT_DIM,
        hidden_dim: int = config.GAT_HIDDEN_DIM,
        num_heads: int = config.GAT_NUM_HEADS,
        dropout: float = config.GAT_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 1. Input projection: Linear(3 → hidden_dim)
        self.input_proj = nn.Linear(in_feats, hidden_dim, bias=True)

        # 2. GATConv layer 1: (hidden_dim → hidden_dim × num_heads)
        #    allow_zero_in_degree=True because self-loops are added externally
        self.gat = dglnn.GATConv(
            in_feats=hidden_dim,
            out_feats=hidden_dim,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            activation=F.elu,
            allow_zero_in_degree=True,
        )

        self.dropout = nn.Dropout(p=dropout)

        # 3. Attention Pooling
        self.pool = dglnn.GlobalAttentionPooling(
            gate_nn=nn.Linear(hidden_dim * num_heads, 1, bias=False)
        )

        # 4. MLP: (hidden_dim × num_heads → 32 → 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        g : dgl.DGLGraph
            Batched or single DGL graph.
            Node feature key: 'feat', shape (total_nodes, in_feats).
        return_embedding : bool
            When True, returns (score, pooled_embedding) so FaceRankNet can
            collect organ embeddings for cross-organ attention.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            score  : shape () or (B,), values in (1, 5)
            embed  : shape (B, hidden_dim × num_heads) — only when return_embedding=True
        """
        h = g.ndata["feat"]                          # (total_nodes, in_feats)

        # Input projection
        h = self.dropout(F.elu(self.input_proj(h)))  # (total_nodes, hidden)

        # GAT: output shape → (total_nodes, num_heads, hidden_dim)
        h = self.gat(g, h)                           # (N, heads, hidden)
        h = h.view(h.shape[0], -1)                   # (N, heads × hidden)

        # Attention pooling: weighted sum of node features based on learned attention
        pooled = self.pool(g, h)                     # (B, heads × hidden)

        # MLP → scalar score per graph
        logit = self.mlp(pooled).squeeze(-1)         # (B,) or scalar
        score = scale_to_score(logit)                # ∈ (1, 5)

        if return_embedding:
            return score, pooled
        return score


# ---------------------------------------------------------------------------
# FaceRankNet
# ---------------------------------------------------------------------------

class FaceRankNet(nn.Module):
    """
    Full FaceRankNet model.

    One OrganGAT per organ.  Organ scores are fused into a global score
    via a softmax-weighted sum over 5 learnable scalar weights.

    Parameters
    ----------
    gat_kwargs : dict | None
        Optional keyword arguments forwarded to OrganGAT.__init__.
    """

    def __init__(self, gat_kwargs: dict | None = None) -> None:
        super().__init__()
        kwargs = gat_kwargs or {}

        # 5 independent OrganGAT modules (one per organ)
        self.organ_gats = nn.ModuleDict(
            {organ: OrganGAT(**kwargs) for organ in ORGAN_ORDER}
        )

        # Learnable fusion weights (5,) — kept for interpretability (organ importance)
        self.fusion_weights = nn.Parameter(
            torch.ones(config.NUM_ORGANS, dtype=torch.float32)
        )

        # Cross-organ attention: captures inter-organ proportion relationships
        embed_dim = config.GAT_HIDDEN_DIM * config.GAT_NUM_HEADS  # 256
        self.cross_organ_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=config.CROSS_ORGAN_HEADS,
            dropout=config.GAT_DROPOUT,
            batch_first=True,
        )

        # Global score MLP: operates on cross-organ attended embeddings
        self.global_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ELU(),
            nn.Dropout(p=config.GAT_DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        subgraphs: dict[str, dgl.DGLGraph],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        subgraphs : dict[str, dgl.DGLGraph]
            Keys must match ``ORGAN_ORDER``.
            Each value is a (possibly batched) DGL graph.

        Returns
        -------
        dict with keys:
            'local_scores'  : dict[str, Tensor] — shape () or (B,) per organ
            'global_score'  : Tensor            — cross-organ attended score, shape (B,)
            'organ_weights' : Tensor            — softmax(w), shape (5,)  [interpretability]
            'attn_weights'  : Tensor            — cross-organ attention map, shape (B, 5, 5)
        """
        local_scores: dict[str, torch.Tensor] = {}
        organ_embeds: list[torch.Tensor] = []

        for organ in ORGAN_ORDER:
            score, embed = self.organ_gats[organ](subgraphs[organ], return_embedding=True)
            local_scores[organ] = score          # () or (B,)
            organ_embeds.append(embed)           # (B, embed_dim)

        # Normalised fusion weights — kept for interpretability output only
        organ_weights = torch.softmax(self.fusion_weights, dim=0)  # (5,)

        # Cross-organ attention: (B, 5, embed_dim)
        embeds = torch.stack(organ_embeds, dim=1)              # (B, 5, embed_dim)
        attn_out, attn_weights = self.cross_organ_attn(
            embeds, embeds, embeds
        )  # attn_out: (B, 5, embed_dim); attn_weights: (B, 5, 5)

        # Mean-pool attended embeddings → global score
        global_embed = attn_out.mean(dim=1)                    # (B, embed_dim)
        global_logit = self.global_mlp(global_embed).squeeze(-1)  # (B,)
        global_score = scale_to_score(global_logit)            # (B,) ∈ (1, 5)

        return {
            "local_scores": local_scores,
            "global_score": global_score,
            "organ_weights": organ_weights,
            "attn_weights": attn_weights,        # (B, 5, 5) — visualizable
        }

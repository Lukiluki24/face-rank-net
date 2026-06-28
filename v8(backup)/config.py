"""
config.py — FaceRankNet
=======================
Single source of truth for every hyperparameter and path constant.
Nothing is hardcoded in any other module — import from here instead.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Dataset paths  (override in run_colab.ipynb after Drive mount)
# ---------------------------------------------------------------------------
DATA_ROOT: Path = Path("data/SCUT-FBP5500")
IMAGE_DIR: Path = DATA_ROOT / "Images"
TRAIN_CSV: Path = DATA_ROOT / "train_labels.csv"
TEST_CSV: Path = DATA_ROOT / "test_labels.csv"

LANDMARK_CACHE_TRAIN: Path = Path("cache/train_landmarks.pkl")
LANDMARK_CACHE_TEST: Path = Path("cache/test_landmarks.pkl")
PSEUDO_LABEL_CACHE: Path = Path("cache/pseudo_labels.pkl")
AVG_FACE_CACHE: Path = Path("cache/avg_face.npy")

CHECKPOINT_PATH: Path = Path("checkpoint_best.pt")

# Set to True to resume training from CHECKPOINT_PATH if it exists.
# Set to False to always start from scratch (overwrites any existing checkpoint).
RESUME_FROM_CHECKPOINT: bool = True

# ---------------------------------------------------------------------------
# Column names expected in CSVs
# ---------------------------------------------------------------------------
COL_FILENAME: str = "Filename"
COL_RATING: str = "Rating"          # holistic beauty score [1, 5]
COL_ETHNICITY: str = "Ethnicity"    # "Asian" | "Caucasian" | etc.

# ---------------------------------------------------------------------------
# Graph / sub-graph
# ---------------------------------------------------------------------------
NODE_FEAT_DIM: int = 6              # (x, y, z, Δx, Δy, Δz) — coords + deviation from avg face

# ---------------------------------------------------------------------------
# OrganGAT architecture
# ---------------------------------------------------------------------------
GAT_HIDDEN_DIM: int = 64
GAT_NUM_HEADS: int = 4
GAT_DROPOUT: float = 0.1

# ---------------------------------------------------------------------------
# FaceRankNet fusion
# ---------------------------------------------------------------------------
ORGAN_NAMES: list[str] = ["left_eye", "right_eye", "nose", "mouth", "jawline"]
NUM_ORGANS: int = len(ORGAN_NAMES)

# Cross-organ attention heads. embed_dim = GAT_HIDDEN_DIM × GAT_NUM_HEADS = 256.
# Must divide evenly into embed_dim (256 / 4 = 64 per head).
CROSS_ORGAN_HEADS: int = 4

# ---------------------------------------------------------------------------
# Fusion mode — how the 5 organ-level outputs combine into global_score
# ---------------------------------------------------------------------------
#   "score_aware"  : global_score = global_mlp( concat(global_embed, local_scores) )
#                    local_scores are concatenated into the global head's input.
#                    Use compute_fusion_sensitivity() in evaluate.py to verify
#                    the head actually uses them.
#
#   "fusion_weight": global_score = Σ softmax(fusion_weights)[i] * local_scores[i]
#                    Drops cross-organ attention + global_mlp entirely.
#                    Pure interpretable weighted sum of local scores.
FUSION_MODE: str = "fusion_weight"

# ---------------------------------------------------------------------------
# Score range enforcement: 4 * sigmoid(x) + 1  →  (1, 5)
# ---------------------------------------------------------------------------
SCORE_MIN: float = 1.0
SCORE_MAX: float = 5.0

# ---------------------------------------------------------------------------
# Loss weights (fixed — no dynamic balancing)
# ---------------------------------------------------------------------------
LRANK_WEIGHT: float = 1.0           # Pairwise ranking loss weight
LDIV_WEIGHT: float = 0.01           # Diversity regularisation weight

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 50
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
TRAIN_VAL_SPLIT: float = 0.9        # fraction of train CSV used for training

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
NUM_WORKERS: int = 0    # must be 0 in Colab — DGL graphs cannot be pickled across workers
PIN_MEMORY: bool = False  # pin_memory only benefits when num_workers > 0

# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------
# Number of negative pairs per anchor in the ranking DataLoader
PAIRS_PER_SAMPLE: int = 3

# Bucket edges used for stratified val split, weighted pair sampler, and
# hard pair sampling. Creates 4 buckets: <2, 2–3, 3–4, >4.
BUCKET_EDGES: tuple[float, ...] = (2.0, 3.0, 4.0)

# Enable WeightedRandomSampler for pair_loader → rebalances anchor rating
# buckets so extreme (jelek/cantik) faces appear ~4× more per epoch.
USE_WEIGHTED_PAIR_SAMPLER: bool = True

# Sampler smoothing: "sqrt" (default, moderate boost ~3.6× for Jelek)
# or "inverse" (aggressive, ~13× boost — overfits when unique count < ~200).
PAIR_SAMPLER_SMOOTHING: str = "sqrt"

# Hard Pair Sampling: bias PairDataset candidate selection toward partners
# in distant rating buckets so each anchor sees more extreme contrasts per
# epoch (Jelek vs Cantik). Random pairing gives ~0.5% such pairs.
USE_HARD_PAIR_SAMPLING: bool = True

# ---------------------------------------------------------------------------
# Pair filtering (H2 consistency)
# ---------------------------------------------------------------------------
# Pseudo-label margin filter — only train L_rank on organ pairs where the
# pseudo-score gap is *confident* (above noise floor). Setting > 0 drops
# noisy near-tie pairs, leaving only high-confidence orderings for ranking.
RANK_PSEUDO_MARGIN: float = 0.3

# Minimum holistic rating gap for H2 pair filter. Pairs where
# rating_a - rating_b < this value are skipped — near-tie holistic
# ratings produce noisy L_rank gradients. Default 0.5 (half a scale step).
MIN_PAIR_RATING_GAP: float = 0.5

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
# Landmark jitter — small Gaussian noise added to (x, y, z) on each
# __getitem__ call. After centroid-normalization landmark coords typically
# fall in ~[-1, 1], so σ=0.003 ≈ 0.3% noise (well below MediaPipe error).
AUGMENT_JITTER: bool = True
JITTER_STD: float = 0.003

# ---------------------------------------------------------------------------
# Logging / display
# ---------------------------------------------------------------------------
LOG_EVERY_N_BATCHES: int = 50

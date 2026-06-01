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

# Cross-organ attention heads (B2). embed_dim = GAT_HIDDEN_DIM × GAT_NUM_HEADS = 256.
# Must divide evenly into embed_dim (256 / 4 = 64 per head).
CROSS_ORGAN_HEADS: int = 4

# ---------------------------------------------------------------------------
# Score range enforcement: 4 * sigmoid(x) + 1  →  (1, 5)
# ---------------------------------------------------------------------------
SCORE_MIN: float = 1.0
SCORE_MAX: float = 5.0

# ---------------------------------------------------------------------------
# GradNorm (Chen et al. 2018)
# ---------------------------------------------------------------------------
GRADNORM_ALPHA: float = 0.5
NUM_TASKS: int = 2                  # L_reg, L_rank  (L_div handled separately)
LDIV_WEIGHT: float = 0.01           # Fixed weight for diversity regularisation

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 50
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
TRAIN_VAL_SPLIT: float = 0.9       # fraction of train CSV used for training

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

# ---------------------------------------------------------------------------
# Imbalance handling (Step 1 — class-balanced sampling + landmark jitter)
# ---------------------------------------------------------------------------
# Enable WeightedRandomSampler for pair_loader → rebalances anchor rating
# buckets so extreme (jelek/cantik) faces appear ~4× more per epoch.
USE_WEIGHTED_PAIR_SAMPLER: bool = True

# Bucket edges used by the sampler (creates 4 buckets: <2, 2–3, 3–4, >4).
PAIR_SAMPLER_BUCKET_EDGES: tuple[float, ...] = (2.0, 3.0, 4.0)

# Sampler smoothing: "sqrt" (default, moderate boost ~3.6× for Jelek)
# or "inverse" (aggressive, ~13× boost — overfits when unique count < ~200).
PAIR_SAMPLER_SMOOTHING: str = "sqrt"

# ---------------------------------------------------------------------------
# L_rank tuning
# ---------------------------------------------------------------------------
# Number of epochs L_rank is frozen (gradient = 0) so L_reg establishes a
# stable regression baseline first. After this epoch L_rank activates AND
# GradNorm L0 is recaptured (otherwise L0 = 1e-8 from frozen state corrupts
# the loss ratio → GradNorm balancing is broken for the rest of training).
RANK_FREEZE_EPOCHS: int = 10

# Linear ramp-up duration AFTER freeze ends. L_rank scale goes 0 → 1 over
# this many epochs to avoid gradient shock. L0 is reset at the end of warmup
# when L_rank reaches its final magnitude.
RANK_WARMUP_EPOCHS: int = 10

# Pseudo-label margin filter — only train L_rank on organ pairs where the
# pseudo-score gap is *confident* (above noise floor). Setting > 0 drops
# noisy near-tie pairs, leaving only high-confidence orderings for ranking.
# Recommended 0.2-0.4 given pseudo-label Spearman ρ ≈ 0.57.
RANK_PSEUDO_MARGIN: float = 0.3

# Landmark jitter — small Gaussian noise added to (x, y, z) on each
# __getitem__ call. After centroid-normalization landmark coords typically
# fall in ~[-1, 1], so σ=0.003 ≈ 0.3% noise (well below MediaPipe error).
AUGMENT_JITTER: bool = True
JITTER_STD: float = 0.003

# ---------------------------------------------------------------------------
# Logging / display
# ---------------------------------------------------------------------------
LOG_EVERY_N_BATCHES: int = 50

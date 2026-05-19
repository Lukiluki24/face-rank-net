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
NODE_FEAT_DIM: int = 3              # (x, y, z) per landmark

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

# ---------------------------------------------------------------------------
# Score range enforcement: 4 * sigmoid(x) + 1  →  (1, 5)
# ---------------------------------------------------------------------------
SCORE_MIN: float = 1.0
SCORE_MAX: float = 5.0

# ---------------------------------------------------------------------------
# GradNorm (Chen et al. 2018)
# ---------------------------------------------------------------------------
GRADNORM_ALPHA: float = 1.5
NUM_TASKS: int = 3                  # L_reg, L_rank, L_div

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
PAIRS_PER_SAMPLE: int = 1

# ---------------------------------------------------------------------------
# Logging / display
# ---------------------------------------------------------------------------
LOG_EVERY_N_BATCHES: int = 50

"""
app_dgf.py — Modal deployment for FaceRankNet-DGF training.

DeepGeoFusion variant:
  - Delaunay topology per organ (anatomically meaningful edges)
  - 5D edge features: [θ_ij, y_ij, γ_ij, α_ij, β_ij]
  - Edge-aware GAT attention (edge geometry enters attention score)

Uses a SEPARATE checkpoint volume (frn-checkpoints-dgf) so the baseline
experiment is fully preserved and resumable at any time.

Usage:
    modal run modal/app_dgf.py                     # fresh DGF run (50 epochs)
    modal run modal/app_dgf.py --epochs 80         # more epochs
    modal run modal/app_dgf.py --resume            # resume DGF checkpoint
    modal run --detach modal/app_dgf.py            # detached (survives disconnect)

    modal run modal/app_dgf.py --download          # download best DGF checkpoint
    modal run modal/app_dgf.py --download-to my_ckpt.pt

Baseline (original, unaffected):
    modal run modal/app.py --resume                # resumes from frn-checkpoints

Volumes:
    frn-data            (shared with baseline)  — CSV labels
    frn-cache           (shared with baseline)  — landmarks, pseudo-labels, avg_face
    frn-checkpoints-dgf (DGF-only)             — DGF checkpoint_best.pt
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image  (identical to app.py — same dependency stack)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgles2-mesa",
        "libegl1",
        "libopengl0",
    )
    .pip_install(
        "torch==2.3.0",
        "numpy>=2.0",
        "pandas",
        "scipy",
        "scikit-learn",
        "tqdm",
        "psutil",
        "networkx",
        "requests",
        "pyyaml",
        "pydantic",
        "mediapipe",
        "opencv-python-headless",
        "Pillow",
    )
    .pip_install(
        "dgl",
        find_links="https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html",
        extra_options="--no-deps",
    )
    # Baseline model code (train.py, dataset.py, loss.py, etc.)
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "model"),
        remote_path="/root/face_rank_net",
        ignore=["*.ipynb", "__pycache__", "*.pyc"],
    )
    # DGF overrides: config.py, preprocessing.py, model.py
    # Placed at a separate path; sys.path inserts it BEFORE face_rank_net
    # so Python finds the DGF versions first on import.
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "model_dgf"),
        remote_path="/root/face_rank_net_dgf",
        ignore=["__pycache__", "*.pyc"],
    )
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
# frn-data and frn-cache are SHARED with the baseline app (read-only usage
# here — same CSV, landmarks, pseudo-labels, avg_face).
# frn-checkpoints-dgf is SEPARATE so DGF and baseline checkpoints never
# overwrite each other.
# ---------------------------------------------------------------------------
data_vol  = modal.Volume.from_name("frn-data",             create_if_missing=True)
cache_vol = modal.Volume.from_name("frn-cache",            create_if_missing=True)
ckpt_vol  = modal.Volume.from_name("frn-checkpoints-dgf",  create_if_missing=True)

DATA_DIR  = "/data"
CACHE_DIR = "/cache"
CKPT_DIR  = "/checkpoints"

app = modal.App("face-rank-net-dgf", image=image)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    volumes={
        DATA_DIR:  data_vol,
        CACHE_DIR: cache_vol,
        CKPT_DIR:  ckpt_vol,
    },
    timeout=60 * 60 * 6,  # 6 h ceiling
)
def train_remote(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    resume: bool = False,  # default False: DGF architecture != baseline checkpoint
) -> dict:
    import sys

    # DGF overrides (config, preprocessing, model) must shadow baseline.
    # Insert model_dgf/ FIRST so its versions are found before model/.
    sys.path.insert(0, "/root/face_rank_net_dgf")
    sys.path.insert(1, "/root/face_rank_net")

    import config  # resolves to model_dgf/config.py (USE_DELAUNAY=True)

    # --- Patch paths ---
    config.AVG_FACE_CACHE       = Path(CACHE_DIR) / "avg_face.npy"
    config.LANDMARK_CACHE_TRAIN = Path(CACHE_DIR) / "train_landmarks.pkl"
    config.LANDMARK_CACHE_TEST  = Path(CACHE_DIR) / "test_landmarks.pkl"
    config.PSEUDO_LABEL_CACHE   = Path(CACHE_DIR) / "pseudo_labels.pkl"
    config.CHECKPOINT_PATH      = Path(CKPT_DIR)  / "checkpoint_best.pt"

    from train import train as train_fn  # from model/ (unchanged)

    train_fn(
        train_csv=str(Path(DATA_DIR) / "train_labels.csv"),
        test_csv=str(Path(DATA_DIR) / "test_labels.csv"),
        landmark_cache_train=str(config.LANDMARK_CACHE_TRAIN),
        landmark_cache_test=str(config.LANDMARK_CACHE_TEST),
        pseudo_labels_path=str(config.PSEUDO_LABEL_CACHE),
        avg_face_path=str(config.AVG_FACE_CACHE),
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_path=str(config.CHECKPOINT_PATH),
        resume=resume,
    )

    ckpt_vol.commit()

    import torch

    ckpt = torch.load(config.CHECKPOINT_PATH, map_location="cpu")
    return {
        "best_pcc":   float(ckpt["best_pcc"]),
        "best_epoch": int(ckpt["epoch"]),
        "checkpoint_path": str(config.CHECKPOINT_PATH),
        "experiment": "dgf",
    }


# ---------------------------------------------------------------------------
# Download DGF checkpoint to local machine
# ---------------------------------------------------------------------------
@app.function(volumes={CKPT_DIR: ckpt_vol})
def fetch_checkpoint() -> bytes:
    return (Path(CKPT_DIR) / "checkpoint_best.pt").read_bytes()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    resume: bool = False,
    download: bool = False,
    download_to: str = "checkpoint_dgf_best.pt",
):
    print(
        f"[DGF] Submitting training job — "
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, resume={resume}"
    )
    print(
        "[DGF] Checkpoint volume: frn-checkpoints-dgf  "
        "(baseline frn-checkpoints untouched)"
    )

    result = train_remote.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        resume=resume,
    )

    print("Training complete:")
    print(f"  experiment  : {result['experiment']}")
    print(f"  best PCC    : {result['best_pcc']:.4f}")
    print(f"  best epoch  : {result['best_epoch']}")
    print(f"  checkpoint  : {result['checkpoint_path']}  (volume: frn-checkpoints-dgf)")

    if download:
        print(f"Downloading DGF checkpoint → {download_to} ...")
        blob = fetch_checkpoint.remote()
        Path(download_to).write_bytes(blob)
        print(f"Saved {len(blob) / 1e6:.1f} MB to {download_to}")

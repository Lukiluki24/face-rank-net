"""
app.py — Modal deployment for FaceRankNet training.

Replaces Colab Cell 8: runs `train.train()` on a Modal A100 with the dataset
and pre-computed caches mounted from persistent Modal Volumes.

Local entrypoint:
    modal run modal/app.py                       # default 50 epochs
    modal run modal/app.py --epochs 80           # override epoch count
    modal run modal/app.py --no-resume           # ignore existing checkpoint

Detached (long jobs survive client disconnect):
    modal run --detach modal/app.py

Before the first run, populate the volumes via `modal/upload_cache.py`.
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# DGL is installed with --no-deps to avoid downgrading torch (same trick as
# Colab Cell 1). Versions match what the Colab notebook uses successfully.
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
    # DGL wheel from their own index (PyPI only has CPU/old versions).
    # torch-2.3 + CUDA 12.1 matches the base image's torch 2.3.0 install.
    .pip_install(
        "dgl",
        find_links="https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html",
        extra_options="--no-deps",
    )
    # Bundle the project's model/ directory into the container.
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "model"),
        remote_path="/root/face_rank_net",
        ignore=["*.ipynb", "__pycache__", "*.pyc"],
    )
)

# ---------------------------------------------------------------------------
# Volumes (persistent across runs)
# ---------------------------------------------------------------------------
#   /data        — train_labels.csv, test_labels.csv (and images if needed)
#   /cache       — landmark .pkl files, pseudo_labels.pkl, avg_face.npy
#   /checkpoints — checkpoint_best.pt
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("frn-data", create_if_missing=True)
cache_vol = modal.Volume.from_name("frn-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("frn-checkpoints", create_if_missing=True)

DATA_DIR = "/data"
CACHE_DIR = "/cache"
CKPT_DIR = "/checkpoints"

app = modal.App("face-rank-net", image=image)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    volumes={
        DATA_DIR: data_vol,
        CACHE_DIR: cache_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=60 * 60 * 6,  # 6 h ceiling
)
def train_remote(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    resume: bool = True,
) -> dict:
    import sys

    sys.path.insert(0, "/root/face_rank_net")

    import config  # noqa: E402

    # Patch config paths to point at the mounted volumes BEFORE importing train,
    # because train.py reads several config constants at import time.
    config.AVG_FACE_CACHE = Path(CACHE_DIR) / "avg_face.npy"
    config.LANDMARK_CACHE_TRAIN = Path(CACHE_DIR) / "train_landmarks.pkl"
    config.LANDMARK_CACHE_TEST = Path(CACHE_DIR) / "test_landmarks.pkl"
    config.PSEUDO_LABEL_CACHE = Path(CACHE_DIR) / "pseudo_labels.pkl"
    config.CHECKPOINT_PATH = Path(CKPT_DIR) / "checkpoint_best.pt"

    from train import train as train_fn  # noqa: E402

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

    # Persist the checkpoint volume so the next run can resume / download.
    ckpt_vol.commit()

    import torch

    ckpt = torch.load(config.CHECKPOINT_PATH, map_location="cpu")
    return {
        "best_pcc": float(ckpt["best_pcc"]),
        "best_epoch": int(ckpt["epoch"]),
        "checkpoint_path": str(config.CHECKPOINT_PATH),
    }


# ---------------------------------------------------------------------------
# Download the trained checkpoint to the local machine
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
    resume: bool = True,
    download: bool = False,
    download_to: str = "checkpoint_best.pt",
):
    print(
        f"Submitting training job — "
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, resume={resume}"
    )
    result = train_remote.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        resume=resume,
    )
    print("Training complete:")
    print(f"  best PCC : {result['best_pcc']:.4f}")
    print(f"  best epoch: {result['best_epoch']}")
    print(f"  checkpoint: {result['checkpoint_path']}  (in volume 'frn-checkpoints')")

    if download:
        print(f"Downloading checkpoint → {download_to} ...")
        blob = fetch_checkpoint.remote()
        Path(download_to).write_bytes(blob)
        print(f"Saved {len(blob) / 1e6:.1f} MB to {download_to}")

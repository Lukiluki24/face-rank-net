"""
app.py — Modal deployment for FaceRankNet training.

Replaces Colab Cell 8: runs `train.train()` on a Modal A100 with the dataset
and pre-computed caches mounted from persistent Modal Volumes.

Typical workflow
----------------
1. Upload caches + checkpoint from Colab:
       python modal/upload_cache.py --checkpoint modal/checkpoint_best.pt

2. Run training (resumes from uploaded checkpoint):
       modal run modal/app.py

3. For long runs that survive client disconnect:
       modal run --detach modal/app.py

4. Download best checkpoint after training:
       modal run modal/app.py --download

CLI flags
---------
    --epochs INT        Total epochs (default 50)
    --batch-size INT    Batch size (default 32; A100 can handle 64+)
    --lr FLOAT          Learning rate (default 1e-3)
    --weight-decay FLOAT
    --no-resume         Force training from scratch (ignore checkpoint)
    --download          Download checkpoint_best.pt after training finishes
    --download-to PATH  Local path for downloaded checkpoint (default: checkpoint_best.pt)
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
    use_augmented_train: bool = True,
) -> dict:
    import sys

    sys.path.insert(0, "/root/face_rank_net")

    import config  # noqa: E402

    # Patch config paths to point at the mounted volumes BEFORE importing train,
    # because train.py reads several config constants at import time.
    config.AVG_FACE_CACHE          = Path(CACHE_DIR) / "avg_face.npy"
    config.LANDMARK_CACHE_TRAIN    = Path(CACHE_DIR) / "train_landmarks.pkl"
    config.LANDMARK_CACHE_TEST     = Path(CACHE_DIR) / "test_landmarks.pkl"
    config.PSEUDO_LABEL_CACHE      = Path(CACHE_DIR) / "pseudo_labels.pkl"
    config.CHECKPOINT_PATH         = Path(CKPT_DIR)  / "checkpoint_best.pt"

    # Switch the training set + landmark cache between original and
    # pseudo-label-time MixUp augmented bundles. Test set always stays clean.
    aug_csv   = Path(DATA_DIR)  / "aug_train_labels.csv"
    aug_cache = Path(CACHE_DIR) / "aug_train_landmarks.pkl"
    if use_augmented_train and aug_csv.exists() and aug_cache.exists():
        train_csv            = str(aug_csv)
        landmark_cache_train = str(aug_cache)
        dataset_tag = "synthaug (real + synthetic tail)"
    else:
        if use_augmented_train:
            print("⚠ augmented files missing — falling back to original cache.")
        train_csv            = str(Path(DATA_DIR)  / "train_labels.csv")
        landmark_cache_train = str(config.LANDMARK_CACHE_TRAIN)
        dataset_tag = "original (no synthaug)"

    # Log active config so Modal dashboard shows what's running.
    print("=" * 60)
    print("FaceRankNet — Modal A100 training")
    print(f"  dataset       : {dataset_tag}")
    print(f"  train CSV     : {train_csv}")
    print(f"  train cache   : {landmark_cache_train}")
    print(f"  pseudo-labels : {config.PSEUDO_LABEL_CACHE}")
    print(f"  epochs        : {num_epochs}")
    print(f"  batch_size    : {batch_size}")
    print(f"  lr            : {lr}")
    print(f"  GRADNORM_ALPHA       : {config.GRADNORM_ALPHA}")
    print(f"  AUGMENT_JITTER       : {config.AUGMENT_JITTER}  std={config.JITTER_STD}")
    print(f"  USE_WEIGHTED_PAIR    : {config.USE_WEIGHTED_PAIR_SAMPLER}")
    print(f"  USE_INVERSE_FREQ_REG : {config.USE_INVERSE_FREQ_L_REG}  (LDS)")
    print(f"  USE_HARD_PAIR_SAMPL  : {config.USE_HARD_PAIR_SAMPLING}  (HPS)")
    print(f"  USE_WITHIN_BUCKET_MIX: {config.USE_WITHIN_BUCKET_MIXUP}  (Paradigma A)")
    print(f"  resume        : {resume}")
    print(f"  checkpoint    : {config.CHECKPOINT_PATH}")
    ckpt_exists = config.CHECKPOINT_PATH.exists()
    if ckpt_exists:
        import torch
        ckpt_meta = torch.load(config.CHECKPOINT_PATH, map_location="cpu")
        print(f"  → resuming from epoch {ckpt_meta['epoch']}, "
              f"best_pcc={ckpt_meta['best_pcc']:.4f}, "
              f"lambdas={[round(l, 3) for l in ckpt_meta.get('lambdas', [])]}")
    else:
        print("  → no checkpoint found, starting from scratch")
    print("=" * 60)

    from train import train as train_fn  # noqa: E402

    train_fn(
        train_csv=train_csv,
        test_csv=str(Path(DATA_DIR) / "test_labels.csv"),
        landmark_cache_train=landmark_cache_train,
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
    result = {
        "best_pcc":   float(ckpt["best_pcc"]),
        "best_epoch": int(ckpt["epoch"]),
        "checkpoint": str(config.CHECKPOINT_PATH),
    }
    print("\n" + "=" * 60)
    print(f"Training done — best PCC={result['best_pcc']:.4f} at epoch {result['best_epoch']}")
    print("=" * 60)
    return result


# ---------------------------------------------------------------------------
# Download the trained checkpoint to the local machine
# ---------------------------------------------------------------------------
@app.function(volumes={CKPT_DIR: ckpt_vol})
def fetch_checkpoint() -> bytes:
    return (Path(CKPT_DIR) / "checkpoint_best.pt").read_bytes()


# ---------------------------------------------------------------------------
# Graceful stop: create STOP file in checkpoint volume
# Run from a second terminal: modal run modal/app.py::stop_training
# ---------------------------------------------------------------------------
@app.function(volumes={CKPT_DIR: ckpt_vol})
def stop_training() -> str:
    stop_path = Path(CKPT_DIR) / "STOP"
    stop_path.touch()
    ckpt_vol.commit()
    return f"STOP file created at {stop_path} — training will halt after current epoch."


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
    use_augmented_train: bool = True,
    download: bool = False,
    download_to: str = "checkpoint_best.pt",
):
    print(
        f"Submitting training job — "
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"resume={resume}, augmented={use_augmented_train}"
    )
    result = train_remote.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        resume=resume,
        use_augmented_train=use_augmented_train,
    )
    print("Training complete:")
    print(f"  best PCC : {result['best_pcc']:.4f}")
    print(f"  best epoch: {result['best_epoch']}")
    print(f"  checkpoint: {result['checkpoint']}  (in volume 'frn-checkpoints')")

    if download:
        print(f"Downloading checkpoint → {download_to} ...")
        blob = fetch_checkpoint.remote()
        Path(download_to).write_bytes(blob)
        print(f"Saved {len(blob) / 1e6:.1f} MB to {download_to}")

"""
train.py — FaceRankNet
=======================
Full training loop with:
  - Adam optimiser (lr from config, weight_decay from config)
  - GradNorm dynamic loss weighting
  - Per-epoch validation: PCC, MAE, DPD printed to stdout
  - Checkpoint saved to config.CHECKPOINT_PATH on best validation PCC
  - tqdm progress bars for epochs and batches

Reproducibility seeds are set at the top of this script.

Usage (from Colab Cell 7):
    %run train.py --train_csv data/.../train.csv \
                  --test_csv  data/.../test.csv  \
                  --landmark_cache_train cache/train_landmarks.pkl \
                  --landmark_cache_test  cache/test_landmarks.pkl  \
                  --pseudo_labels        cache/pseudo_labels.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import (
    FaceDataset,
    PairDataset,
    make_face_loader,
    make_pair_loader,
    make_weighted_pair_loader,
)
from evaluate import run_full_evaluation, validate_local_scores
from loss import GradNorm, l_div, l_rank, l_reg
from model import FaceRankNet
from pseudo_labels import load_avg_face, load_pseudo_labels, validate_pseudo_label_quality

# ---- Reproducibility ----
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
dgl.seed(config.SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    logger.info("Using device: %s", dev)
    return dev


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    train_csv: str,
    test_csv: str,
    landmark_cache_train: str,
    landmark_cache_test: str,
    pseudo_labels_path: str,
    avg_face_path: str = str(config.AVG_FACE_CACHE),
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    weight_decay: float = config.WEIGHT_DECAY,
    checkpoint_path: str = str(config.CHECKPOINT_PATH),
    resume: bool = config.RESUME_FROM_CHECKPOINT,
) -> FaceRankNet:
    """
    Full training procedure.

    Parameters
    ----------
    train_csv             : Path to training CSV (Filename, Rating[, Ethnicity]).
    test_csv              : Path to test/validation CSV.
    landmark_cache_train  : Path to train landmark .pkl cache.
    landmark_cache_test   : Path to test landmark .pkl cache.
    pseudo_labels_path    : Path to pseudo-labels .pkl.
    num_epochs, batch_size, lr, weight_decay: hyper-parameters.
    checkpoint_path       : Where to save best model.

    Returns
    -------
    FaceRankNet — the trained model (best checkpoint loaded).
    """
    device = get_device()

    # ---- Load caches ----
    with open(landmark_cache_train, "rb") as f:
        coords_train: dict = pickle.load(f)
    with open(landmark_cache_test, "rb") as f:
        coords_test: dict = pickle.load(f)

    pseudo_labels = load_pseudo_labels(pseudo_labels_path)

    # ---- Diagnostic: validate pseudo-label quality before training ----
    train_df = pd.read_csv(train_csv)
    holistic_ratings = dict(zip(
        train_df[config.COL_FILENAME].tolist(),
        train_df[config.COL_RATING].astype(float).tolist(),
    ))
    validate_pseudo_label_quality(pseudo_labels, holistic_ratings)

    # avg_face is computed from training set only — safe to use for both
    # train and test node features (no leakage).
    avg_face = load_avg_face(avg_face_path)

    # ---- Datasets ----
    # Step 1: enable landmark jitter on train set only.  Combined with the
    # WeightedRandomSampler below, this gives minority-class faces fresh
    # geometric variation each time they are resampled.
    # Sprint 2: LDS rating weights + within-bucket MixUp activated on TRAIN
    # only (test stays clean for unbiased evaluation).
    train_face_ds = FaceDataset(
        train_csv, coords_train, pseudo_labels, avg_face=avg_face,
        augment_jitter=config.AUGMENT_JITTER,
        compute_lds_weights=config.USE_INVERSE_FREQ_L_REG,
        mixup_within_bucket=config.USE_WITHIN_BUCKET_MIXUP,
    )
    test_face_ds  = FaceDataset(test_csv,  coords_test,  avg_face=avg_face)

    pair_ds = PairDataset(
        train_face_ds,
        hard_pair_sampling=config.USE_HARD_PAIR_SAMPLING,
    )

    logger.info(
        "Datasets — train faces: %d | pairs: %d | test: %d",
        len(train_face_ds),
        len(pair_ds),
        len(test_face_ds),
    )

    val_loader = make_face_loader(test_face_ds, shuffle=False, batch_size=batch_size)

    # ---- Model ----
    model = FaceRankNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("FaceRankNet — trainable parameters: %d", total_params)

    # ---- Optimiser (task parameters only; GradNorm handles λ separately) ----
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- GradNorm ----
    gradnorm = GradNorm(model, num_tasks=config.NUM_TASKS, alpha=config.GRADNORM_ALPHA)

    # ---- Checkpoint: resume or start fresh ----
    best_pcc: float = -1.0
    start_epoch: int = 1

    ckpt_path_obj = Path(checkpoint_path)
    if resume and ckpt_path_obj.exists():
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)

        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_pcc   = ckpt["best_pcc"]
            start_epoch = ckpt["epoch"] + 1

            # Restore GradNorm lambdas
            if "lambdas" in ckpt:
                gradnorm.lambdas = torch.tensor(
                    ckpt["lambdas"], dtype=torch.float32, device=device
                )

            if start_epoch > num_epochs:
                logger.info(
                    "Checkpoint is already at epoch %d — nothing left to train.",
                    ckpt["epoch"],
                )
                return model

            logger.info(
                "Resumed at epoch %d / %d  (best PCC so far: %.4f)",
                start_epoch,
                num_epochs,
                best_pcc,
            )
        except RuntimeError as e:
            logger.warning(
                "Checkpoint architecture mismatch — starting fresh.\n"
                "  Reason: %s", e
            )
            best_pcc    = -1.0
            start_epoch = 1
    else:
        if resume and not ckpt_path_obj.exists():
            logger.info(
                "resume=True but no checkpoint found at '%s' — starting fresh.",
                checkpoint_path,
            )

    # ---- Graceful stop: create a file named "STOP" to halt after current epoch ----
    stop_flag = Path(checkpoint_path).parent / "STOP"

    # ---- Epoch loop ----
    # Reset GradNorm L0 at the first epoch where L_rank reaches full magnitude
    # (after freeze + warmup). Without this, L0_rank stays clamped at 1e-8
    # (captured while L_rank was scaled to zero) and the loss ratio explodes,
    # corrupting GradNorm's task balancing for the rest of training.
    rank_l0_reset_epoch = config.RANK_FREEZE_EPOCHS + config.RANK_WARMUP_EPOCHS + 1

    for epoch in range(start_epoch, num_epochs + 1):
        if stop_flag.exists():
            logger.info("STOP file detected — halting training after epoch %d.", epoch - 1)
            stop_flag.unlink()
            break

        if epoch == rank_l0_reset_epoch:
            gradnorm.reset_L0()
            logger.info(
                "  ↻ GradNorm L0 reset at epoch %d (L_rank now at full scale).",
                epoch,
            )

        # Resample pairs every epoch so the model sees different (A, B)
        # combinations and doesn't overfit to a fixed set of pair orderings.
        pair_ds._pairs = pair_ds._build_pairs()
        if config.USE_WEIGHTED_PAIR_SAMPLER:
            # Step 1: rebalance anchor rating bucket each epoch (extreme
            # faces appear ~4× more often → cures regression-to-mean).
            pair_loader = make_weighted_pair_loader(
                pair_ds, batch_size=batch_size,
                bucket_edges=config.PAIR_SAMPLER_BUCKET_EDGES,
                smoothing=config.PAIR_SAMPLER_SMOOTHING,
            )
        else:
            pair_loader = make_pair_loader(pair_ds, shuffle=True, batch_size=batch_size)

        model.train()
        total_loss_accum = 0.0
        n_batches = 0

        pbar = tqdm(pair_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch",
                    miniters=50, mininterval=30, dynamic_ncols=False)
        for batch_a, batch_b, organ_mask in pbar:
            # Move to device
            sg_a = {k: v.to(device) for k, v in batch_a["subgraphs"].items()}
            sg_b = {k: v.to(device) for k, v in batch_b["subgraphs"].items()}
            ratings_a = batch_a["ratings"].to(device)
            ratings_b = batch_b["ratings"].to(device)
            organ_mask = organ_mask.to(device)
            # LDS weights (None when flag is off — falls back to plain MSE)
            if config.USE_INVERSE_FREQ_L_REG:
                w_a = batch_a["rating_weights"].to(device)
                w_b = batch_b["rating_weights"].to(device)
            else:
                w_a = None
                w_b = None

            optimizer.zero_grad()

            # ---- Forward pass ----
            out_a = model(sg_a)
            out_b = model(sg_b)

            global_pred_a = out_a["global_score"]
            global_pred_b = out_b["global_score"]
            local_a = out_a["local_scores"]
            local_b = out_b["local_scores"]

            # ---- Three losses ----
            # Use both faces for regression — previously only face A was used,
            # wasting 50% of the available ground-truth signal each batch.
            loss_reg = (
                l_reg(global_pred_a, ratings_a, weights=w_a)
                + l_reg(global_pred_b, ratings_b, weights=w_b)
            ) / 2
            loss_rank = l_rank(local_a, local_b, organ_mask)
            loss_div  = l_div(local_a)

            # A1: L_rank warmup — freeze for RANK_FREEZE_EPOCHS, then linearly
            # ramp 0→1 over RANK_WARMUP_EPOCHS. Hard on/off would shock GradNorm.
            freeze_end = config.RANK_FREEZE_EPOCHS
            warmup_end = freeze_end + config.RANK_WARMUP_EPOCHS
            if epoch <= freeze_end:
                rank_scale = 0.0
            elif epoch <= warmup_end:
                rank_scale = (epoch - freeze_end) / config.RANK_WARMUP_EPOCHS
            else:
                rank_scale = 1.0
            losses = [loss_reg, rank_scale * loss_rank]

            # L_div is negative (−Var) so it must NOT enter GradNorm,
            # which only handles positive losses. Add it with a fixed weight.

            # ---- GradNorm: weighted total loss (L_reg + L_rank only) ----
            total_loss = gradnorm.update(losses, optimizer)
            total_loss = total_loss + config.LDIV_WEIGHT * loss_div

            # ---- Backward + step ----
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_accum += total_loss.item()
            n_batches += 1

            if n_batches % config.LOG_EVERY_N_BATCHES == 0:
                pbar.set_postfix(
                    loss=f"{total_loss_accum / n_batches:.4f}",
                    lam=f"{gradnorm.lambdas.tolist()}",
                )

        avg_loss = total_loss_accum / max(n_batches, 1)

        # ---- Validation ----
        metrics = run_full_evaluation(model, val_loader, device)
        pcc = metrics["pcc"]
        mae = metrics["mae"]
        dpd = metrics["dpd"]

        epoch_summary = (
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"PCC={pcc:.4f} | MAE={mae:.4f} | DPD={dpd:.4f}"
        )
        tqdm.write(epoch_summary)
        logger.info(epoch_summary)

        # ---- Checkpoint ----
        if pcc > best_pcc:
            best_pcc = pcc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pcc": best_pcc,
                    "lambdas": gradnorm.lambdas.tolist(),
                },
                checkpoint_path,
            )
            logger.info("  ✓ Checkpoint saved (PCC=%.4f) → %s", best_pcc, checkpoint_path)

    # ---- Load best checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        "Training complete. Best PCC=%.4f (epoch %d).",
        ckpt["best_pcc"],
        ckpt["epoch"],
    )

    # ---- Local score validity ----
    logger.info("Running local score validity check (Spearman ρ) …")
    valid = validate_local_scores(model, train_face_ds, device)
    logger.info("Local score validity: %s", "PASS ✓" if valid else "FAIL ✗")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FaceRankNet — training script"
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--landmark_cache_train", required=True)
    parser.add_argument("--landmark_cache_test", required=True)
    parser.add_argument("--pseudo_labels", required=True)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_PATH))
    parser.add_argument(
        "--resume", action="store_true", default=config.RESUME_FROM_CHECKPOINT,
        help="Resume training from --checkpoint if it exists (default: per config.py)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Always start training from scratch, ignoring any existing checkpoint.",
    )
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        landmark_cache_train=args.landmark_cache_train,
        landmark_cache_test=args.landmark_cache_test,
        pseudo_labels_path=args.pseudo_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )

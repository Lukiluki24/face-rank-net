"""
train.py — FaceRankNet
=======================
Clean training loop:
  - Adam optimiser
  - Fixed loss weights:  L_reg + LRANK_WEIGHT * L_rank + LDIV_WEIGHT * L_div
  - Stratified train/val split (config.TRAIN_VAL_SPLIT)
  - Checkpoint saved on best VAL PCC
  - Test set evaluated exactly once after training
  - Per-epoch metrics → results.csv
  - Training curves → training_curves.png

Usage (from Colab Cell 8):
    run_training(
        train_csv=..., test_csv=...,
        landmark_cache_train=..., landmark_cache_test=...,
        pseudo_labels_path=..., avg_face_path=...,
        checkpoint_path=..., resume=True,
    )
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from dataset import (
    FaceDataset,
    PairDataset,
    _bucket_of,
    make_face_loader,
    make_pair_loader,
    make_weighted_pair_loader,
)
from evaluate import (
    compute_fusion_sensitivity,
    print_fusion_sensitivity,
    run_full_evaluation,
    validate_local_scores,
)
from loss import l_div, l_rank, l_reg
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
# Stratified val split
# ---------------------------------------------------------------------------

def _stratified_val_split(
    df: pd.DataFrame,
    train_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train_df, val_df) stratified by rating bucket."""
    bucket_labels = [
        _bucket_of(float(r), config.BUCKET_EDGES)
        for r in df[config.COL_RATING].tolist()
    ]
    train_df, val_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=bucket_labels,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def _save_training_curves(
    history: list[dict],
    test_metrics: dict[str, float] | None,
    best_epoch: int,
    save_path: str,
) -> None:
    """Save a training-curve figure: loss | val PCC | val MAE."""
    if not history:
        return

    import matplotlib.pyplot as plt

    epochs   = [h["epoch"]   for h in history]
    losses   = [h["loss"]    for h in history]
    val_pccs = [h["val_pcc"] for h in history]
    val_maes = [h["val_mae"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FaceRankNet — Training Curves", fontsize=14)

    ax = axes[0]
    ax.plot(epochs, losses, "b-", lw=1.5, label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(epochs, val_pccs, "g-", lw=1.5, label="Val PCC")
    if best_epoch in epochs:
        ax.axvline(best_epoch, color="red", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    ax.axhline(0.70, color="gray", ls=":", lw=1, label="Target 0.70")
    if test_metrics is not None:
        ax.axhline(
            test_metrics["pcc"], color="orange", ls="-.", lw=1.5,
            label=f'Test PCC={test_metrics["pcc"]:.4f}',
        )
    ax.set_xlabel("Epoch"); ax.set_ylabel("PCC")
    ax.set_title("Validation PCC")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(epochs, val_maes, "r-", lw=1.5, label="Val MAE")
    ax.axhline(0.36, color="gray", ls=":", lw=1, label="Target 0.36")
    if test_metrics is not None:
        ax.axhline(
            test_metrics["mae"], color="orange", ls="-.", lw=1.5,
            label=f'Test MAE={test_metrics["mae"]:.4f}',
        )
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE")
    ax.set_title("Validation MAE")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", save_path)


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
    on_new_best: "callable | None" = None,
) -> FaceRankNet:
    """
    Full training procedure with fixed loss weights.

    Total loss per batch:
        L = L_reg + LRANK_WEIGHT * L_rank + LDIV_WEIGHT * L_div

    Both faces in each pair contribute to L_reg.

    Returns
    -------
    FaceRankNet — the trained model (best val-PCC checkpoint loaded).
    """
    device = get_device()

    # ---- Load caches ----
    with open(landmark_cache_train, "rb") as f:
        coords_train: dict = pickle.load(f)
    with open(landmark_cache_test, "rb") as f:
        coords_test: dict = pickle.load(f)

    pseudo_labels = load_pseudo_labels(pseudo_labels_path)
    avg_face = load_avg_face(avg_face_path)

    # ---- Carve val split from train_csv ----
    full_train_df = pd.read_csv(train_csv)
    train_df, val_df = _stratified_val_split(
        full_train_df,
        train_frac=config.TRAIN_VAL_SPLIT,
        seed=config.SEED,
    )
    logger.info(
        "Val split: %d train / %d val (stratified, TRAIN_VAL_SPLIT=%.2f)",
        len(train_df), len(val_df), config.TRAIN_VAL_SPLIT,
    )

    # ---- Pseudo-label quality diagnostic ----
    holistic_ratings = dict(zip(
        train_df[config.COL_FILENAME].tolist(),
        train_df[config.COL_RATING].astype(float).tolist(),
    ))
    validate_pseudo_label_quality(pseudo_labels, holistic_ratings)

    # ---- Datasets ----
    train_face_ds = FaceDataset(
        train_df, coords_train, pseudo_labels, avg_face=avg_face,
        augment_jitter=config.AUGMENT_JITTER,
    )
    val_face_ds  = FaceDataset(val_df,  coords_train, avg_face=avg_face)
    test_face_ds = FaceDataset(test_csv, coords_test, avg_face=avg_face)

    pair_ds = PairDataset(
        train_face_ds,
        hard_pair_sampling=config.USE_HARD_PAIR_SAMPLING,
    )

    logger.info(
        "Datasets — train faces: %d | pairs: %d | val: %d | test: %d",
        len(train_face_ds), len(pair_ds), len(val_face_ds), len(test_face_ds),
    )

    val_loader  = make_face_loader(val_face_ds,  shuffle=False, batch_size=batch_size)
    test_loader = make_face_loader(test_face_ds, shuffle=False, batch_size=batch_size)

    # ---- Model ----
    model = FaceRankNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("FaceRankNet — trainable parameters: %d", total_params)

    # ---- Optimiser ----
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Checkpoint: resume or start fresh ----
    best_pcc: float  = -1.0
    best_epoch: int  = 0
    start_epoch: int = 1

    ckpt_path_obj = Path(checkpoint_path)
    if resume and ckpt_path_obj.exists():
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)

        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_pcc    = ckpt["best_pcc"]
            best_epoch  = ckpt.get("best_epoch", ckpt["epoch"])
            start_epoch = ckpt["epoch"] + 1

            if start_epoch > num_epochs:
                logger.info(
                    "Checkpoint is already at epoch %d — nothing left to train.",
                    ckpt["epoch"],
                )
                return model

            logger.info(
                "Resumed at epoch %d / %d  (best val PCC so far: %.4f)",
                start_epoch, num_epochs, best_pcc,
            )
        except RuntimeError as e:
            logger.warning(
                "Checkpoint architecture mismatch — starting fresh.\n"
                "  Reason: %s", e
            )
            best_pcc    = -1.0
            best_epoch  = 0
            start_epoch = 1
    else:
        if resume and not ckpt_path_obj.exists():
            logger.info(
                "resume=True but no checkpoint found at '%s' — starting fresh.",
                checkpoint_path,
            )

    # ---- Graceful stop flag ----
    stop_flag = Path(checkpoint_path).parent / "STOP"

    history: list[dict] = []

    for epoch in range(start_epoch, num_epochs + 1):
        if stop_flag.exists():
            logger.info("STOP file detected — halting training after epoch %d.", epoch - 1)
            stop_flag.unlink()
            break

        # Resample pairs every epoch so the model sees different (A, B) combos.
        pair_ds._pairs = pair_ds._build_pairs()
        if config.USE_WEIGHTED_PAIR_SAMPLER:
            pair_loader = make_weighted_pair_loader(
                pair_ds, batch_size=batch_size,
                bucket_edges=config.BUCKET_EDGES,
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
            sg_a = {k: v.to(device) for k, v in batch_a["subgraphs"].items()}
            sg_b = {k: v.to(device) for k, v in batch_b["subgraphs"].items()}
            ratings_a  = batch_a["ratings"].to(device)
            ratings_b  = batch_b["ratings"].to(device)
            organ_mask = organ_mask.to(device)

            optimizer.zero_grad()

            out_a = model(sg_a)
            out_b = model(sg_b)

            global_pred_a = out_a["global_score"]
            global_pred_b = out_b["global_score"]
            local_a = out_a["local_scores"]
            local_b = out_b["local_scores"]

            loss_reg = (
                l_reg(global_pred_a, ratings_a)
                + l_reg(global_pred_b, ratings_b)
            ) / 2
            loss_rank = l_rank(local_a, local_b, organ_mask)
            loss_div  = l_div(local_a)

            total_loss = (
                loss_reg
                + config.LRANK_WEIGHT * loss_rank
                + config.LDIV_WEIGHT * loss_div
            )
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_accum += total_loss.item()
            n_batches += 1

            if n_batches % config.LOG_EVERY_N_BATCHES == 0:
                pbar.set_postfix(loss=f"{total_loss_accum / n_batches:.4f}")

        avg_loss = total_loss_accum / max(n_batches, 1)

        # ---- Validation on val set ----
        val_metrics = run_full_evaluation(model, val_loader, device)
        val_pcc = val_metrics["pcc"]
        val_mae = val_metrics["mae"]
        val_dpd = val_metrics["dpd"]

        history.append({
            "epoch":   epoch,
            "loss":    avg_loss,
            "val_pcc": val_pcc,
            "val_mae": val_mae,
            "val_dpd": val_dpd,
        })

        epoch_summary = (
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val_PCC={val_pcc:.4f} | val_MAE={val_mae:.4f} | val_DPD={val_dpd:.4f}"
        )
        tqdm.write(epoch_summary)
        logger.info(epoch_summary)

        # ---- Checkpoint on val PCC only ----
        if val_pcc > best_pcc:
            best_pcc   = val_pcc
            best_epoch = epoch
            torch.save(
                {
                    "epoch":                epoch,
                    "best_epoch":           best_epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pcc":             best_pcc,
                    "best_val_mae":         val_mae,
                    "best_val_dpd":         val_dpd,
                },
                checkpoint_path,
            )
            logger.info("  Checkpoint saved (val_PCC=%.4f) -> %s", best_pcc, checkpoint_path)
            if on_new_best is not None:
                on_new_best()

    # ---- Load best checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
    logger.info(
        "Training complete. Best val PCC=%.4f (epoch %d).",
        ckpt["best_pcc"], best_epoch,
    )

    # ---- Evaluate on test set EXACTLY ONCE ----
    logger.info("Evaluating on test set (once, best checkpoint) ...")
    test_metrics = run_full_evaluation(model, test_loader, device)
    logger.info(
        "Test  PCC=%.4f | MAE=%.4f | DPD=%.4f",
        test_metrics["pcc"], test_metrics["mae"], test_metrics["dpd"],
    )

    # ---- Append to results.csv ----
    results_csv_path = Path(checkpoint_path).parent / "results.csv"
    write_header = not results_csv_path.exists()
    with open(results_csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow([
                "best_epoch", "total_epochs",
                "val_pcc", "val_mae", "val_dpd",
                "test_pcc", "test_mae", "test_dpd",
            ])
        writer.writerow([
            best_epoch,
            num_epochs,
            f"{ckpt['best_pcc']:.4f}",
            f"{ckpt.get('best_val_mae', float('nan')):.4f}",
            f"{ckpt.get('best_val_dpd', float('nan')):.4f}",
            f"{test_metrics['pcc']:.4f}",
            f"{test_metrics['mae']:.4f}",
            f"{test_metrics['dpd']:.4f}",
        ])
    logger.info("Results written to %s", results_csv_path)

    # ---- Local score validity ----
    logger.info("Running local score validity check (Spearman rho) ...")
    valid = validate_local_scores(model, train_face_ds, device)
    logger.info("Local score validity: %s", "PASS" if valid else "FAIL")

    # ---- Fusion sensitivity: does global_score actually use local_scores? ----
    logger.info("Computing fusion sensitivity on test set ...")
    sens_report = compute_fusion_sensitivity(model, test_loader, device)
    print_fusion_sensitivity(sens_report)

    # ---- Save training curves ----
    curves_path = str(Path(checkpoint_path).parent / "training_curves.png")
    _save_training_curves(
        history=history,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        save_path=curves_path,
    )

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceRankNet — training script")
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
        help="Resume training from --checkpoint if it exists.",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Always start from scratch, ignoring any existing checkpoint.",
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

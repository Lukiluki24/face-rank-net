# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FaceRankNet** — a facial beauty prediction system using Graph Attention Networks on 468 MediaPipe 3D landmarks. Runs in **Google Colab** (T4 GPU). No local training environment. All `.py` files are uploaded to Google Drive and copied into the Colab VM at runtime.

Dataset: SCUT-FBP5500 (5500 images, holistic beauty scores [1,5], Asian + Caucasian ethnicities).

## Architecture

### Data Flow
```
Raw image → MediaPipe (468 3D landmarks) → normalize (centroid + inter-ocular)
→ 5 organ sub-graphs → OrganGAT per organ → GlobalAttentionPooling
→ MLP → local_score ∈ (1,5) → softmax-weighted fusion → global_score
```

### Key Design Decisions
- **Node features**: 6-dim `(x, y, z, Δx, Δy, Δz)` — raw coords + deviation from Universal Average Face. Requires `avg_face` passed to `FaceDataset`, else falls back to 3-dim (breaks model).
- **Organ sub-graphs**: fully connected + self-loops. 5 organs: `left_eye, right_eye, nose, mouth, jawline`. Eyebrows merged into eye sub-graphs.
- **Score range**: enforced via `4 * sigmoid(x) + 1` → output always ∈ (1, 5).
- **L_div weight** (`LDIV_WEIGHT=0.01`) is fixed, not GradNorm-managed. Only `L_reg` and `L_rank` have dynamic λ.

### Module Responsibilities
| File | Role |
|------|------|
| `config.py` | Single source of truth — all hyperparams and paths |
| `organ_indices.py` | MediaPipe landmark index lists per organ |
| `preprocessing.py` | Extract landmarks, normalize, build DGL sub-graphs, cache to `.pkl` |
| `pseudo_labels.py` | Compute Universal Average Face, per-organ RMSE → pseudo-scores [1,5] |
| `dataset.py` | `FaceDataset`, `PairDataset` (for L_rank), collate functions |
| `model.py` | `OrganGAT`, `FaceRankNet` |
| `loss.py` | `l_reg`, `l_rank`, `l_div`, `GradNorm` |
| `train.py` | Full training loop with GradNorm, checkpoint save/resume |
| `evaluate.py` | PCC, MAE, DPD, local score validity check |

## Colab Workflow

Cell execution order (never skip):
1. **Cell 1** — install packages (`--no-deps` for DGL to avoid breaking torch)
2. **Cell 2** — import check (DGL + CUDA)
3. **Cell 3** — mount Drive, copy `.py` files, parse dataset CSV (adds `Ethnicity` column from filename prefix: `A*`=Asian, `C*`=Caucasian)
4. **Cell 4** — extract landmarks (~20-40 min, cached to Drive)
5. **Cell 5** — compute per-ethnicity avg faces + pseudo-labels (H1 refinement)
6. **Cell 6** — build datasets/dataloaders (must pass `avg_face` to `FaceDataset`)
7. **Cell 7** — model dry-run
8. **Cell 8** — training (patches `config.AVG_FACE_CACHE` before calling `run_training`)

After Cell 1 completes, **do not re-run it** — no runtime restart needed with `--no-deps`.

## Critical Gotchas

- **`avg_face` must be passed to `FaceDataset`** — without it, nodes get 3-dim features and the model's `Linear(6→64)` crashes.
- **`train.py` reads `config.AVG_FACE_CACHE` at call time** — patch `config.AVG_FACE_CACHE = pathlib.Path(AVG_FACE_PATH)` in the notebook before calling `run_training()`.
- **`config.py` paths are relative** (e.g., `cache/avg_face.npy`) — they don't exist in Colab. Always override via the notebook.
- **`NUM_WORKERS=0`** required in Colab — DGL graphs can't be pickled across workers.
- **`PairDataset._build_pairs()`** applies H2 consistency filter: skips pairs where `holistic_a <= holistic_b`.

## Pseudo-label Generation (WSL via Averageness Hypothesis)

Pseudo-score formula: `score = 5 - 4 * (RMSE_organ / max_RMSE_organ)`

H1 refinement: compute separate avg faces per ethnicity (`compute_ethnicity_avg_faces()`), pass `avg_face_map` + `ethnicity_map` to `compute_all_pseudo_labels()`.

H2 refinement: `PairDataset._build_pairs()` only creates pair (A,B) if `holistic_rating(A) > holistic_rating(B)`.

Diagnostic: `validate_pseudo_label_quality()` prints Spearman ρ between mean pseudo-score and holistic rating. Target ρ > 0.2.

## Hyperparameters (config.py)

```
GAT_HIDDEN_DIM=64, GAT_NUM_HEADS=4, GAT_DROPOUT=0.1
BATCH_SIZE=32, NUM_EPOCHS=50, LR=1e-3, WEIGHT_DECAY=1e-4
PAIRS_PER_SAMPLE=3, LDIV_WEIGHT=0.01, GRADNORM_ALPHA=1.5
```

## Training Metrics to Monitor

- **PCC** (Pearson): target >0.70 by epoch 50
- **MAE**: target <0.36 by epoch 50
- **λ_rank**: should stay near 1.0 or decrease; if it climbs past 1.2, L_rank is conflicting
- **Spearman ρ** at Cell 5: if < 0.2, pseudo-labels are noisy — H1+H2 needed

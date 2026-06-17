# FaceRankNet — Experiment Summary

**Purpose.** This document is a single source of truth for the FaceRankNet pipeline **as actually implemented and trained**, used as the baseline to revise [FaceRankNet_Paper.md](FaceRankNet_Paper.md). The paper draft was written before several ablations (GradNorm removal, LDS removal, MixUp removal, pseudo-label switch to beauty-axis, L_rank toggle) landed on the codebase, so several sections of the paper no longer match the code or the trained checkpoint. Every divergence is flagged inline with `[PAPER MISMATCH — fix §X]` so the revision pass is mechanical.

Last training run: best validation epoch 49 / 50. Test set: PCC **0.6222**, MAE **0.4410**, DPD **0.0855**.

---

## 1. Dataset & Splits

- **Dataset:** SCUT-FBP5500 — 5,500 frontal facial images, 60-rater averaged holistic beauty scores ∈ [1,5], two ethnicities (Asian, Caucasian) × two genders.
- **Ethnicity column:** derived from filename prefix at CSV load time (`A*` → Asian, `C*` → Caucasian).
- **Splits:**
  - Standard 80/20 train/test (4,400 / 1,100 images) — matches benchmark.
  - **Additional 90/10 stratified train/val carved from the 4,400 train portion** (`TRAIN_VAL_SPLIT=0.9`, stratified by rating bucket). Effective splits: 3,960 train / 440 val / 1,100 test.
  - Model selection: checkpoint with best **validation** PCC; test set is evaluated exactly once at the end.

> `[PAPER MISMATCH — fix §IV.A]` The paper claims only "80% training / 20% testing." Add the stratified 90/10 train/val carve; without a val set there is no model-selection signal to discuss.

---

## 2. Preprocessing Pipeline

1. **Face detection:** MediaPipe Face Mesh → 468 three-dimensional landmark coordinates `(X_i, Y_i, Z_i)` per face. MediaPipe assigns consistent anatomical indices.
2. **Coordinate normalization:**
   - Centroid centering: subtract mean of all 468 landmarks.
   - Inter-ocular scale normalization: divide by Euclidean distance between landmarks 33 ↔ 263 (outer eye corners).
3. **Node features — 6-dimensional `(x, y, z, Δx, Δy, Δz)`:** the second triple is the **per-node deviation from the Universal Average Face** (mean of all training landmarks). The deviation channel is what feeds the Averageness signal into every node and the input linear layer is `Linear(6 → 64)`. If `avg_face` is not passed to `FaceDataset`, nodes fall back to 3D and the model crashes.
4. **Sub-graph partitioning — 5 organs:** `left_eye, right_eye, nose, mouth, jawline`. Eyebrows are merged into the respective eye sub-graphs. Each sub-graph is **fully connected with self-loops** (`dgl.add_self_loop`).

> `[PAPER MISMATCH — fix §III.B(1)]` Paper describes only 3D coords. The 6D node feature with `Δ` from the Universal Average Face is missing and is a core part of how the Averageness hypothesis is wired into the model.

> `[PAPER MISMATCH — fix §III.B(3)]` Paper lists "Left Eye + Left Brow, Right Eye + Right Brow, ...". Phrasing is fine, but make sure the eyes are described as **two independent sub-graphs** (left and right), not one combined "eyes" graph — the code builds two.

---

## 3. Weakly Supervised Pseudo-Label Generation

Two methods exist in `model/pseudo_labels.py`; **the training pipeline uses the second**.

| Method | Function | Used in training? |
|---|---|---|
| Classic RMSE-from-average | `compute_all_pseudo_labels` | **No** |
| Beauty-axis projection | `compute_all_pseudo_labels_beauty_axis` | **Yes (default in Cell 5)** |

### Beauty-axis projection (production method)

For each ethnicity group `e`:

1. `population_mean_e = mean(train_coords | ethnicity = e)`.
2. `beauty_prototype_e = mean(top-30% rated training faces | ethnicity = e)`.
3. `beauty_axis_e = beauty_prototype_e − population_mean_e` — a per-coordinate vector pointing from the average to the "attractive end" of the population.
4. For each training face `f` and organ `o`:
   - Restrict `beauty_axis_e` to the landmark indices of organ `o`.
   - Compute scalar projection of `(face_coords[o] − population_mean_e[o])` onto that organ axis.
5. Per-organ projections are **percentile-ranked across the training set** and remapped: `pseudo_score = 1 + 4 · percentile_rank ∈ [1, 5]`.

**H1 refinement (per-ethnicity prototypes & axes)** means Asian faces and Caucasian faces are each compared against their own group's prototype/axis. This is the entire H1 mechanism — there is no separate H1 module elsewhere.

**Validation gate:** at Cell 5 we compute Spearman ρ between mean pseudo-score (averaged over organs) and the holistic ground-truth rating. Target ρ > 0.2 before training proceeds.

> `[PAPER MISMATCH — fix §III.D(1)]` Paper derives pseudo-labels as `5 − 4·(MSE/max MSE)` — pure RMSE distance to the average face. **Rewrite this entire subsection** to describe beauty-axis projection with per-ethnicity prototype/axis (H1), percentile normalization, and the Spearman ρ ≥ 0.2 validation gate. The current MSE formula is from an earlier version and no longer matches the code.

> `[PAPER MISMATCH — fix §II.C]` Related-work section can stay but should now mention "beauty-axis projection" rather than implying generic MSE-from-average WSL.

---

## 4. Model Architecture

### 4.1 `OrganGAT` (one per organ)

| Layer | Spec |
|---|---|
| Input projection | `Linear(6 → 64)` |
| Graph conv | `GATConv(64 → 64, num_heads=4, residual=True, activation=ELU)` → flatten heads → 256-d node embeddings |
| Pooling | **DGL `GlobalAttentionPooling`** with a learnable gate `Linear(256 → 1)` |
| Read-out MLP | `Linear(256 → 32) → ELU → Dropout(0.1) → Linear(32 → 1)` |
| Score range | `local_score = 4 · sigmoid(x) + 1 ∈ (1, 5)` |

> `[PAPER MISMATCH — fix §III.C(3)]` Paper writes pooling as `β_i = softmax(w^T h'_i)`. The implementation uses DGL `GlobalAttentionPooling` (gated attention pooling). Replace the formula with the gated form or simply describe it as "gated attention pooling implemented via DGL `GlobalAttentionPooling`".

### 4.2 Fusion → Global Score (production = `fusion_weight`)

`config.py:76` sets `FUSION_MODE = "fusion_weight"`. This is the architecture that produced the reported results.

$$
\hat{y}_{\text{global}} = \sum_{i=1}^{5} \mathrm{softmax}(w_i) \cdot \hat{y}_{\text{organ}_i}
$$

Properties:
- Output is mathematically guaranteed in `[1, 5]` because each `local_score ∈ (1,5)` and weights sum to 1.
- **No cross-organ attention, no global MLP** — these layers are dropped entirely.
- `organ_weights = softmax(w_i)` is returned and is **directly interpretable** as the relative aesthetic contribution of each organ.

A second mode `score_aware` exists in the codebase (cross-organ `MultiheadAttention` + `MLP(concat(global_embed, local_scores))`) but **was not used to produce the reported results** and is treated only as an ablated alternative.

> `[PAPER MISMATCH — fix §III.C(5)]` The paper already describes the softmax-weighted fusion correctly. Keep this section. **Do not** add cross-organ attention to the methodology — the trained checkpoint does not use it.

> `[PAPER MISMATCH — fix Abstract / §III]` Any mention of "cross-organ attention" as part of the proposed architecture should be removed; the production model is the pure interpretable weighted sum.

---

## 5. Loss Function (Fixed Weights — No GradNorm)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reg}} + 1.0 \cdot \mathcal{L}_{\text{rank}} + 0.01 \cdot \mathcal{L}_{\text{div}}
$$

Weights are **fixed constants** from `config.py` (`LRANK_WEIGHT=1.0`, `LDIV_WEIGHT=0.01`). GradNorm, LDS, and MixUp were removed in commit `9dbd52f` (clean baseline). An L_rank=0 ablation (`405d88a`) confirmed that disabling the ranking loss degrades quality, so L_rank was re-enabled in `36c3b82`.

### 5.1 Anchor regression

$$\mathcal{L}_{\text{reg}} = \mathrm{MSE}(\hat{y}_{\text{global}}, y_{\text{gt}})$$

### 5.2 Feature-level pairwise ranking (with two filters)

$$
\mathcal{L}_{\text{rank}} = \frac{1}{|\text{kept}|}\sum_{o}\sum_{(A,B)} m_{o}(A,B) \cdot \log\!\bigl(1 + \exp\bigl(\hat{y}_{o}(B) - \hat{y}_{o}(A)\bigr)\bigr)
$$

where the **per-organ mask** $m_o(A,B) = \mathbb{1}\bigl[\hat{y}^{psc}_o(A) - \hat{y}^{psc}_o(B) > \tau\bigr]$, $\tau =$ `RANK_PSEUDO_MARGIN = 0.3`. Implemented with `log1p(exp(·))` for numerical stability.

### 5.3 Diversity + boundary regularization

$$
\mathcal{L}_{\text{div}} = -\mathrm{Var}_{o}\bigl(\hat{y}_o\bigr) + \mathrm{ReLU}\bigl(1.2 - \hat{y}_o\bigr) + \mathrm{ReLU}\bigl(\hat{y}_o - 4.8\bigr)
$$

The variance term spreads scores across organs; the two hinge terms keep local scores away from sigmoid saturation (the `(1,5)` open interval has no gradient at the boundaries).

> `[PAPER MISMATCH — fix §III.D(3.c)]` Paper has only `L_div = −Var(ŷ_organ)`. Add the boundary-hinge terms; they are not cosmetic — without them the L_rank gradient pushes extreme scores toward saturation.

> `[PAPER MISMATCH — fix §III.D(2)]` Paper says only "pairs $(A,B)$ are sampled from the training set." Add the two filters: pseudo-margin mask `τ=0.3` (per-organ) and the holistic-gap filter `MIN_PAIR_RATING_GAP=0.5` (per-pair, applied at pair construction — see §6).

> `[PAPER MISMATCH — fix §III.D(4)]` **Delete the entire "Backpropagation and Dynamic Gradient Normalization (GradNorm)" subsection.** GradNorm is no longer used. Replace with a short paragraph: "Loss weights are fixed at $\lambda_1{=}\lambda_2{=}1.0$, $\lambda_3{=}0.01$. An ablation with $\lambda_2{=}0$ confirmed the ranking term is required for organ-level fidelity (Section [ablation])." Reference [28] (GradNorm) can be dropped if no longer cited elsewhere.

---

## 6. Pair Construction & Sampling

The ranking task uses a dedicated `PairDataset` whose pairs are **rebuilt every epoch** in `train.py` (`pair_ds._pairs = pair_ds._build_pairs()`).

- **H2 consistency filter** at pair construction: drop pairs where `|rating_A − rating_B| < MIN_PAIR_RATING_GAP = 0.5`. Eliminates near-tie pairs whose ordering is dominated by inter-rater noise.
- `PAIRS_PER_SAMPLE = 3` negatives per anchor.
- **Weighted pair sampler** (`USE_WEIGHTED_PAIR_SAMPLER=True`, `PAIR_SAMPLER_SMOOTHING="sqrt"`): rebalances anchor rating buckets so extreme faces appear ~3.6× more per epoch (otherwise Jelek/Cantik buckets are starved).
- **Hard pair sampler** (`USE_HARD_PAIR_SAMPLING=True`): biases candidate selection toward distant rating buckets so each anchor sees more extreme contrasts.
- **Landmark jitter augmentation**: `AUGMENT_JITTER=True`, σ=0.003 after inter-ocular normalization (≈0.3% noise, well below MediaPipe localization error).

> `[PAPER MISMATCH — fix §III.D(2)]` Add a short paragraph describing the H2 filter, the weighted+hard sampler combination, and the jitter augmentation. These materially shape training (rare-bucket coverage, contrast strength) and should appear in Methods or Experimental Setup.

---

## 7. Training Configuration

| Setting | Value |
|---|---|
| Framework | PyTorch + DGL |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR scheduler | none |
| Batch size | 32 |
| Epochs | 50 |
| Loss weights | $\lambda_1{=}1.0$ ($L_{\text{reg}}$), $\lambda_2{=}1.0$ ($L_{\text{rank}}$), $\lambda_3{=}0.01$ ($L_{\text{div}}$) — fixed |
| Model selection | best validation PCC |
| Fusion mode | `fusion_weight` |
| Pair sampling | weighted (sqrt) + hard, rebuilt every epoch |
| Augmentation | landmark jitter σ=0.003 |
| `NUM_WORKERS` | 0 (DGL graphs cannot pickle across workers in Colab) |

---

## 8. Evaluation Protocol

- **Global accuracy:** PCC (Pearson) and MAE on the holistic ground-truth score.
- **Demographic Parity Difference:** $\mathrm{DPD} = |\mathrm{MAE}_{\text{Asian}} - \mathrm{MAE}_{\text{Caucasian}}|$.
- **Local score validity:** per-organ Spearman ρ between predicted `local_score` and the beauty-axis pseudo-score. The check passes only if all five organ correlations are positive.
- **Fusion sensitivity diagnostic** (`compute_fusion_sensitivity`, `evaluate.py`): autograd-based ratio $\sum_i \lvert \partial \hat{y}_{\text{global}} / \partial \hat{y}_{\text{organ}_i} \rvert$ vs. $\dim_{\text{embed}} \cdot \overline{\lvert \partial \hat{y}_{\text{global}} / \partial \text{global\_embed}_k \rvert}$ — verifies the global head actually uses the local scores. In **`fusion_weight` mode this diagnostic is trivially passed by construction** (the global is a closed-form weighted sum of local scores; `global_embed` is `None`). It is meaningful only in `score_aware` mode.
- **Organ importance distribution:** the learned `softmax(w_i)` weights are reported as the per-organ aesthetic contribution.

> `[PAPER MISMATCH — fix §III.E]` Add the local-validity Spearman ρ check. State that "Organ Importance Distribution" is read directly off `softmax(w_i)` because the production model is `fusion_weight`. The fusion sensitivity diagnostic can be mentioned as a check used during ablations of the `score_aware` variant but is not applied to the production model.

---

## 9. Results

### 9.1 Top-line metrics

| Split | PCC | MAE | DPD |
|---|---|---|---|
| Validation (best epoch 49 / 50) | 0.6745 | 0.4232 | 0.0510 |
| **Test (held-out 1,100 images)** | **0.6222** | **0.4410** | **0.0855** |

### 9.2 Per-ethnicity breakdown — H1 validation (test set)

| Group | n | PCC | MAE |
|---|---|---|---|
| Asian | 822 | 0.672 | 0.419 |
| Caucasian | 278 | 0.454 | 0.505 |

The per-ethnicity per-prototype design (H1) keeps DPD small in absolute terms (0.086), but the Caucasian subgroup is both noisier and lower-correlated. The smaller `n` (278 vs 822) is a contributing factor and should be acknowledged.

### 9.3 Signed-error per rating bucket (test set)

| Bucket | n | Mean signed error (pred − GT) | Reading |
|---|---|---|---|
| Jelek (1–2) | 43 | **+0.60** | severely over-predicts low scores |
| Rata-rata (2–3) | 610 | +0.03 | well-calibrated |
| Cukup (3–4) | 321 | −0.31 | under-predicts |
| Cantik (4–5) | 126 | **−0.82** | severely under-predicts high scores |

The pattern is classic **regression to the mean**: extreme buckets are systematically pulled inward. The KDE plot of predicted vs ground-truth distribution shows the predicted distribution is narrower than ground truth, with no mass above ~4.3 even though the test set has 126 faces with rating ≥ 4.

### 9.4 Training dynamics

- Training loss is still drifting downward at epoch 50 (final ≈ 0.53). Validation PCC peaks at epoch 49 at 0.6745, val MAE at 0.4153 around epoch 50, with no plateau visible.
- This is **under-fitting, not over-fitting** — a longer schedule or LR warmup/decay should close the gap to the targets in [CLAUDE.md](CLAUDE.md) (PCC > 0.70, MAE < 0.36).

### 9.5 Per-organ qualitative evidence

The four example faces in the organ-scores figure (AM1127, AM1934, AF668, AM756) show clearly **distinct per-organ scores** — they are not numerical copies of the global score. For example:
- AF668 has global 2.75 with all organ scores in [2.6, 3.4], showing balanced contributions.
- AM756 has global 2.34 with organs in [1.1, 2.5], showing the model identifies eyes/jawline as the weak components.
- AM1934 has global 3.93 with organs in [3.6, 5.0], showing the high score is composed across organs with right eye and nose at ceiling.

This is the **main qualitative validation of the part-based design**: the architecture produces interpretable per-organ explanations rather than collapsing into a copy of the global head.

### 9.6 Targets and gap analysis

| Metric | Target ([CLAUDE.md](CLAUDE.md)) | Achieved (test) | Gap |
|---|---|---|---|
| PCC | > 0.70 | 0.6222 | −0.078 |
| MAE | < 0.36 | 0.4410 | +0.081 |
| Spearman ρ (Cell 5 gate) | > 0.20 | — | (was met or training would not have proceeded) |

Both global metrics are short of the targeted values. Likely causes: (i) under-training (val curve still rising), (ii) the regression-to-mean pattern caps PCC since the extreme buckets dominate the long tails of the distribution.

---

## 10. Honest framing for the paper

> `[PAPER MISMATCH — fix Abstract, §I, §III.A]` The current draft says geometric input **"structurally eliminates demographic bias."** The data does not support that strong a claim. Use language like **"structurally reduces"** or **"mitigates"** demographic bias, and back it up with: DPD = 0.0855; Asian PCC 0.672 vs Caucasian PCC 0.454 (a 0.22 gap); Asian MAE 0.419 vs Caucasian MAE 0.505. The bias is reduced relative to pixel-based baselines but is not zero.

> `[PAPER MISMATCH — fix §V Conclusion]` Current conclusion is a stub (`"By …"`). Write the conclusion using actual results — quote test PCC/MAE/DPD, the per-organ interpretability evidence, and explicitly note the regression-to-mean limitation as future work.

> `[PAPER MISMATCH — fix new §IV / §V Results section]` The current paper has no Results section. Add one with the three tables above, the four reference figures, and a short paragraph on the regression-to-mean failure mode (this is the most important qualitative finding to discuss).

---

## 11. Consolidated paper-revision checklist

| Paper section | Action |
|---|---|
| Abstract | Replace "structurally eliminates" → "structurally reduces" bias. Drop any "cross-organ attention" wording. Update reported metrics to test PCC 0.62, MAE 0.44, DPD 0.09. |
| §I Introduction | Same softening of bias claim. |
| §II.C Related Work | Mention "beauty-axis projection" instead of generic MSE WSL. |
| §III.A | Soften "structurally eliminates" claim. |
| §III.B(1) | Add 6D node features (coords + Δ from Universal Average Face). |
| §III.B(3) | Confirm two independent eye sub-graphs (L/R), not one. |
| §III.C(3) | Replace pooling formula with gated attention pooling (DGL `GlobalAttentionPooling`). |
| §III.C(5) | Keep softmax-weighted fusion (matches code). Remove any cross-organ attention wording. |
| §III.D(1) | **Rewrite entirely** — replace MSE pseudo-label formula with beauty-axis projection, per-ethnicity prototype/axis (H1), percentile normalization, Spearman ρ ≥ 0.2 gate. |
| §III.D(2) | Add H2 filter (`MIN_PAIR_RATING_GAP=0.5`), per-organ pseudo margin (`τ=0.3`), weighted+hard pair samplers, landmark jitter augmentation. |
| §III.D(3.c) | Add boundary-hinge terms to $\mathcal{L}_{\text{div}}$. |
| §III.D(4) | **Delete the entire GradNorm subsection.** Replace with short fixed-weight justification + ablation reference. |
| §III.E | Add Spearman ρ local-validity check. Clarify "Organ Importance" comes from `softmax(w_i)` in `fusion_weight` mode. |
| §IV.A | Add stratified 90/10 train/val carve from the 80% train portion. |
| §IV (new Results subsection) | Add the three result tables, four figures, and bias / regression-to-mean discussion. |
| §V Conclusion | Replace the `"By …"` stub with a full conclusion based on actual results. |
| References | Drop [28] GradNorm if no longer cited. |

---

## 12. Verification (run after applying revisions)

1. `python -c "import model.config as c; print(c.FUSION_MODE, c.LRANK_WEIGHT, c.LDIV_WEIGHT, c.NODE_FEAT_DIM)"` → expect `fusion_weight 1.0 0.01 6`.
2. Open [model/loss.py](model/loss.py) — confirm `l_div` contains both `relu(1.2 - …)` and `relu(… - 4.8)` boundary terms.
3. Open `run_colab.ipynb` Cell 5 — confirm it calls `compute_all_pseudo_labels_beauty_axis` (not `compute_all_pseudo_labels`).
4. Re-run [model/evaluate.py](model/evaluate.py) against `checkpoint_best.pt` and confirm test PCC ≈ 0.6222, MAE ≈ 0.4410, DPD ≈ 0.0855 (deterministic up to PyTorch nondeterminism for inference).
5. Regenerate the four figures (error per bucket, per-ethnicity PCC/MAE, organ scores, training curves) from the same checkpoint and visually diff against the attached references.

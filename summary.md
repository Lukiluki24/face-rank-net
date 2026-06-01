# Summary — Perubahan dari Model Initialization ke Versi Saat Ini

Dokumen ini meringkas evolusi FaceRankNet dari commit awal `2c90ba2` ("add: Model Initialization") sampai HEAD (`dab250f`).

## Riwayat Commit

| Commit | Pesan | Tanggal |
|--------|-------|---------|
| `2c90ba2` | add: Model Initialization | 2026-05-19 |
| `87c930b` | fix: fix bug, add features to gat (katanya membantu) | 2026-05-20 |
| `e09059d` | fix: benerin bug training | 2026-05-20 |
| `4b5850c` | fix: benerin dpd | 2026-05-21 |
| `1d52558` | refine: beauty prototype + percentile pseudo-label normalization | 2026-05-22 |
| `ba5ccf2` | Improve pseudo label | 2026-05-22 |
| `197968c` | fix : make collab use cache | 2026-05-22 |
| `dab250f` | run di modal | 2026-05-22 |

## File yang TIDAK Berubah
- [model/organ_indices.py](model/organ_indices.py)
- [model/evaluate.py](model/evaluate.py) — definisi PCC / MAE / DPD / `validate_local_scores` tetap sama

Perubahan arsitektur **ada** (cross-organ attention ditambahkan di [model/model.py](model/model.py)), berbeda dari versi awal di mana fusion hanya softmax-weighted scalar. Sisanya: **fitur input, pseudo-label, loss-weighting, loop training, infrastruktur deployment**.

---

## 1. Konfigurasi — [model/config.py](model/config.py)

| Parameter | Awal (`2c90ba2`) | Sekarang | Alasan |
|-----------|------------------|----------|--------|
| `NODE_FEAT_DIM` | `3` (x, y, z) | `6` (x, y, z, Δx, Δy, Δz) | Tambahan deviasi dari avg face di level input |
| `NUM_TASKS` | `3` (L_reg, L_rank, L_div) | `2` (L_reg, L_rank) | L_div dikeluarkan dari GradNorm karena bernilai negatif |
| `LDIV_WEIGHT` | — | `0.01` | Bobot tetap untuk L_div (di luar GradNorm) |
| `PAIRS_PER_SAMPLE` | `1` | `3` | Lebih banyak pasangan negatif per anchor untuk L_rank |
| `GRADNORM_ALPHA` | `1.5` | `0.5` | Restoring force lebih lembut → menghindari swing λ |
| `CROSS_ORGAN_HEADS` | — | `4` | Heads untuk cross-organ MultiheadAttention (B2) |
| `RESUME_FROM_CHECKPOINT` | — | `True` | Default resume dari checkpoint terakhir |
| `RANK_FREEZE_EPOCHS` | — | `10` | L_rank dibekukan (skala 0) pada N epoch awal agar L_reg stabil |
| `RANK_WARMUP_EPOCHS` | — | `10` | Linear ramp 0→1 untuk L_rank setelah freeze |
| `RANK_PSEUDO_MARGIN` | — | `0.3` | Confidence margin — hanya pair dengan gap pseudo-score > 0.3 yang dipakai L_rank |
| `USE_WEIGHTED_PAIR_SAMPLER` | — | `True` | WeightedRandomSampler untuk rebalance bucket rating ekstrem |
| `PAIR_SAMPLER_BUCKET_EDGES` | — | `(2.0, 3.0, 4.0)` | 4 bucket: jelek, 2–3, 3–4, cantik |
| `PAIR_SAMPLER_SMOOTHING` | — | `"sqrt"` | √-smoothing → ~3.6× boost ke bucket minoritas (lebih aman dari inverse) |
| `AUGMENT_JITTER` | — | `True` | Gaussian noise kecil pada koordinat landmark per `__getitem__` |
| `JITTER_STD` | — | `0.003` | ~0.3% setelah inter-ocular normalization |

---

## 2. Preprocessing — [model/preprocessing.py](model/preprocessing.py)

- `build_subgraph` & `build_all_subgraphs` menerima parameter opsional `avg_face`:
  - `avg_face=None` → fitur node 3-dim (backward compatible).
  - `avg_face` diberikan → fitur node 6-dim: `[koordinat, deviation = coords − avg_face]`.
- Fungsi baru [`build_all_subgraphs_flipped`](model/preprocessing.py#L278): membangun sub-graph untuk wajah ter-flip horizontal (X axis dibalik, label `left_eye` ↔ `right_eye` ditukar agar tetap anatomis benar). Digunakan ketika `augment_flip=True`.

---

## 3. Dataset — [model/dataset.py](model/dataset.py)

### `FaceDataset`
- Parameter baru: `avg_face`, `augment_flip`, `augment_jitter`, `jitter_std`.
- `augment_flip=True` menduplikasi dataset (original + mirrored).
- `augment_jitter=True` menambahkan Gaussian noise (`σ = JITTER_STD`) pada koordinat setiap kali `__getitem__` dipanggil — minoritas yang di-resample oleh WeightedSampler melihat geometri sedikit berbeda tiap epoch.

### `PairDataset` (H2 + confidence margin)
- **H2**: `if rating_a <= rating_b: continue` — hanya pair dengan urutan holistic konsisten.
- **Margin filter**: `organ_mask = (pseudo_a − pseudo_b) > RANK_PSEUDO_MARGIN` — gap di bawah noise floor (`< 0.3`) di-mask 0; L_rank hanya dilatih pada ordering yang confident.

### `make_weighted_pair_loader` (baru)
- WeightedRandomSampler yang me-rebalance anchor rating bucket. Bucket dihitung dari edges `(2.0, 3.0, 4.0)` → 4 kelas (Jelek, 2–3, 3–4, Cantik).
- Bobot: `1 / count` (inverse) atau `1 / √count` (sqrt — default, lebih aman).
- Mengoreksi imbalance SCUT-FBP5500 di mana Jelek (~4.7%) dan Cantik (~11%) sangat under-represented.

---

## 4. Pseudo Labels — [model/pseudo_labels.py](model/pseudo_labels.py)

### a) MSE → RMSE
`compute_organ_mse` return `sqrt(mean(diff²))` — satuan konsisten antar organ.

### b) Beauty Prototype (top-30%)
Fungsi baru [`compute_beauty_prototype`](model/pseudo_labels.py#L147): mean dari **top-k% wajah dengan holistic rating tertinggi** (default 30%). Mengganti "Universal Average Face" yang mencampur cantik + jelek dan menurunkan signal averageness.

### c) Per-Ethnicity Reference (H1)
[`compute_ethnicity_avg_faces`](model/pseudo_labels.py#L250): satu reference face per kelompok etnis (Asian / Caucasian). Bila `holistic_ratings` diberikan, memakai beauty prototype per-etnis; bila tidak, fallback ke population mean per-etnis.

### d) Percentile-Rank Normalization
[`compute_all_pseudo_labels`](model/pseudo_labels.py#L292): Pass 1 kumpulkan RMSE per organ → sortir. Pass 2: `rank = bisect_left(sorted_mse, mse) / n`, lalu `score = clip(5 − 4·rank, 1, 5)`. Distribusi pseudo-score jadi uniform di [1, 5] — tidak ada kompresi dari outlier tunggal.

### e) Beauty Axis Projection (eksperimen alternatif)
[`compute_beauty_axis`](model/pseudo_labels.py#L188), [`project_organ_onto_axis`](model/pseudo_labels.py#L217), [`compute_all_pseudo_labels_beauty_axis`](model/pseudo_labels.py#L386): pseudo-label berdasarkan proyeksi skalar `(face − population_mean) · beauty_axis` di mana `beauty_axis = beauty_prototype − population_mean`. Implementasi referensi untuk Said & Todorov (2011) / DeBruine & Jones (2007) — wajah cantik di sepanjang arah, bukan dekat satu titik prototype.

### f) Quality Diagnostic
[`validate_pseudo_label_quality`](model/pseudo_labels.py#L515): Spearman ρ antara mean pseudo-score per wajah vs holistic rating. Warning bila ρ < 0.2.

---

## 5. Model — [model/model.py](model/model.py)

Berbeda dari versi awal: fusion **bukan lagi** softmax-weighted scalar saja.

**Sekarang**:
- 5 `OrganGAT` menghasilkan `(score, embedding)` per organ.
- Embedding di-stack → `(B, 5, 256)`.
- **Cross-organ MultiheadAttention** (`CROSS_ORGAN_HEADS=4`) menghasilkan attended embeddings.
- Mean pool → `global_mlp(256 → 64 → 1)` → `4·sigmoid + 1` → global_score ∈ (1,5).
- `fusion_weights` parameter dipertahankan **hanya untuk interpretability** (organ importance), bukan untuk menghitung global score.
- `forward()` return: `local_scores`, `global_score`, `organ_weights`, `attn_weights` (B, 5, 5 — visualizable).

---

## 6. Loss — [model/loss.py](model/loss.py)

`GradNorm._get_shared_params` direvisi:
- **Sebelum**: hanya `input_proj` dari organ pertama → cuma menerima gradien dari L_reg.
- **Sekarang**: `Linear(32→1)` dari semua `OrganGAT.mlp` → menerima gradien dari L_reg (via global_score) **dan** L_rank (via local_scores), sesuai prasyarat GradNorm.

Tambahan: [`GradNorm.reset_L0()`](model/loss.py#L181) — memaksa L0 di-recapture di update berikutnya. Dipakai setelah `L_rank` selesai warmup; tanpa reset, L0 yang terekam saat L_rank=0 (clamp ke 1e-8) akan membuat rasio `l_current / L0` meledak dan merusak balancing seterusnya.

---

## 7. Training Loop — [model/train.py](model/train.py)

1. **`avg_face` di-load** dan diteruskan ke train + test dataset.
2. **Diagnostic call** `validate_pseudo_label_quality` sebelum training.
3. **Pair resampling per epoch** — `pair_ds._pairs = pair_ds._build_pairs()` di awal tiap epoch.
4. **WeightedRandomSampler** dibangun ulang tiap epoch (`make_weighted_pair_loader`) bila `USE_WEIGHTED_PAIR_SAMPLER`.
5. **L_rank warmup (A1)**:
   - Epoch 1..`RANK_FREEZE_EPOCHS` (10): `rank_scale = 0` → L_rank dibekukan.
   - Epoch 11..`freeze+warmup` (20): linear ramp 0 → 1.
   - Epoch 21+: `rank_scale = 1`. Pada awal `freeze + warmup + 1` (epoch 21), `gradnorm.reset_L0()` dipanggil.
6. **L_reg pakai kedua wajah**: `(l_reg(pred_a, rating_a) + l_reg(pred_b, rating_b)) / 2`.
7. **L_div dipisah dari GradNorm**:
   ```python
   total_loss = gradnorm.update([loss_reg, rank_scale * loss_rank], optimizer)
   total_loss = total_loss + config.LDIV_WEIGHT * loss_div
   ```
8. **Gradient clipping**: `clip_grad_norm_(max_norm=5.0)` setelah backward.
9. **Resume from checkpoint** (`RESUME_FROM_CHECKPOINT=True`): muat `model_state_dict`, `optimizer_state_dict`, `best_pcc`, `epoch`, dan `lambdas` GradNorm.
10. **Graceful stop**: file `STOP` di samping checkpoint → training berhenti di akhir epoch berjalan.

---

## 8. Modal Deployment — [modal/](modal/)

Migrasi training dari Colab T4 ke Modal A100.

- [modal/app.py](modal/app.py) — baseline app. Volume: `frn-data` (CSV), `frn-cache` (landmarks/pseudo/avg_face), `frn-checkpoints` (checkpoint_best.pt). Patch `config.AVG_FACE_CACHE` / `LANDMARK_CACHE_*` / `PSEUDO_LABEL_CACHE` / `CHECKPOINT_PATH` ke path volume **sebelum** `import train`.
- [modal/app_dgf.py](modal/app_dgf.py) — varian DeepGeoFusion (Delaunay topology + 5D edge features + edge-aware GAT). Bundling tambahan dari direktori `model_dgf/` (sibling project). Checkpoint terpisah di volume `frn-checkpoints-dgf`.
- [modal/diagnostic.py](modal/diagnostic.py) — visualisasi graph topology + edge feature distribution untuk DGF.
- [modal/upload_cache.py](modal/upload_cache.py) — upload `train_landmarks.pkl`, `test_landmarks.pkl`, `pseudo_labels.pkl`, `avg_face.npy`, CSV, dan checkpoint dari lokal ke Modal Volumes.

GPU: `A100-40GB`. Timeout: 6 jam. `--detach` untuk run yang survive disconnect. `stop_training` function men-touch `STOP` di volume untuk halt graceful.

---

## Ringkasan Padat

Tujuh perubahan inti dibanding versi awal:

1. **Node feature 6-dim** dengan deviasi dari avg face.
2. **Pseudo-label** pakai RMSE + **beauty prototype** (top-30% per-etnis) + **percentile rank**. Diagnostic Spearman ρ wajib dijalankan.
3. **PairDataset** di-gate H2 (urutan holistic konsisten) **dan** margin (`RANK_PSEUDO_MARGIN=0.3`).
4. **Cross-organ MultiheadAttention** (4 heads) menggantikan softmax fusion sederhana untuk menghasilkan global_score; `fusion_weights` jadi parameter interpretability saja.
5. **GradNorm** anchor di MLP-tail (bukan input_proj), handle hanya L_reg + L_rank, `α=0.5`, `reset_L0()` setelah warmup. L_div pakai bobot tetap.
6. **Training stabilization**: rank warmup (freeze 10 + ramp 10), pair resampling per epoch, L_reg pakai A+B, gradient clipping, weighted-sampler untuk minoritas rating, jitter & flip augmentation, resume from checkpoint, STOP file untuk graceful halt.
7. **Modal A100 deployment** (`modal/app.py`, `app_dgf.py`, `diagnostic.py`, `upload_cache.py`) — training pindah dari Colab T4 ke Modal dengan volume persistent untuk cache + checkpoint.

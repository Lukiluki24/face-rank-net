# Summary — FaceRankNet: Pipeline & Evolusi Codebase

Dokumen ini berisi dua bagian:
1. **Alur Pipeline Saat Ini** — metode end-to-end yang sedang dipakai (per HEAD `ac45bd7`).
2. **Evolusi Codebase** — apa yang berubah sejak `2c90ba2` ("add: Model Initialization") sampai sekarang, per modul.

---

# Bagian 1 — Alur Pipeline Saat Ini

## 1.1. End-to-End Flow

```
                       ┌─────────────────────────────────────────────────────┐
                       │  Raw Face Image (SCUT-FBP5500, JPEG)                │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 1              │  MediaPipe Face Mesh → 468 landmark (x, y, z)       │
   Landmark Extraction │  → centroid centering + inter-ocular normalization  │
                       │  → coords ∈ ℝ^(468 × 3)  [model/preprocessing.py]   │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 2              │  Reference faces (per ethnicity, H1):               │
   Reference Faces     │   • population_mean_e = mean(all faces in e)        │
                       │   • beauty_prototype_e = mean(top-30% rated in e)   │
                       │  [pseudo_labels.py: compute_universal_average_face, │
                       │   compute_beauty_prototype,                         │
                       │   compute_ethnicity_avg_faces]                      │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 3              │  Beauty axis (per ethnicity, per organ):            │
   Beauty Axis &       │     axis_e = beauty_prototype_e − population_mean_e │
   Pseudo-Label Gen.   │                                                     │
                       │  Untuk tiap wajah / tiap organ:                     │
                       │     dev   = coord_organ − μ_organ                   │
                       │     proj  = (dev · axis_organ) / ‖axis_organ‖       │
                       │     rank  = bisect(sorted_proj, proj) / n           │
                       │     pseudo_score = clip(1 + 4·rank, 1, 5)           │
                       │                                                     │
                       │  Direction: higher projection along beauty axis     │
                       │             → higher score (Said & Todorov 2011;    │
                       │             DeBruine & Jones 2007).                 │
                       │  [pseudo_labels.py:                                 │
                       │   compute_all_pseudo_labels_beauty_axis]            │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 4              │  Diagnostic Spearman ρ(mean pseudo, holistic)       │
   Quality Check       │  Empirically: ρ ≈ 0.57 setelah beauty-axis.         │
                       │  (vs ρ ≈ −0.13 saat proximity-to-population-mean).  │
                       │  [pseudo_labels.py: validate_pseudo_label_quality]  │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 5              │  Build 5 organ sub-graphs PER WAJAH:                │
   Graph Construction  │    nodes: indeks landmark per organ                 │
                       │    edges: fully-connected + self-loops              │
                       │    node feature 6-dim: [x,y,z, Δx,Δy,Δz]            │
                       │    Δ = coord − population_mean                      │
                       │      (avg_face yang disimpan ke disk =              │
                       │       global population_mean dari train set)        │
                       │  [preprocessing.py: build_all_subgraphs]            │
                       │  Optional augmentation:                             │
                       │    • jitter σ=0.003 (per __getitem__)       │ 
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 6              │  PairDataset:                                       │
   Pair Construction   │    (A, B) where rating(A) > rating(B)        ← H2   │
                       │    organ_mask = (pseudo_A − pseudo_B) > 0.3  ← H3   │
                       │  → pair pool DI-REBUILD setiap epoch                │
                       │  [dataset.py: PairDataset]                          │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 7              │  WeightedRandomSampler over 4 rating buckets        │
   Class-Balance       │  {<2, 2–3, 3–4, >4}, weight ∝ 1/√count              │
                       │  → minoritas (Jelek + Cantik) muncul ~3.6× lipat    │
                       │  [dataset.py: make_weighted_pair_loader]            │
                       │                                                     │
                       │  Tail-fix Sprint 2 (semua aktif by default):        │
                       │  • LDS (Label Distribution Smoothing):              │
                       │    per-sample weight = 1/KDE_density(rating)        │
                       │    → tail samples diberi penalti lebih besar di     │
                       │    L_reg (Yang et al. 2021 ICML). bandwidth=0.3     │
                       │  • Hard Pair Sampling (HPS):                        │
                       │    PairDataset memilih kandidat dengan bobot        │
                       │    proporsional terhadap jarak bucket rating.       │
                       │    Jelek↔Cantik pair muncul jauh lebih sering.      │
                       │  • Within-bucket MixUp: OFF by default (ablation).  │
                       │    Paradigma B (MixUp di pseudo-label time via      │
                       │    axis_eth_synthaug) lebih diutamakan.             │
                       │  [config.py, dataset.py, pseudo_labels.py]         │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 8              │  FaceRankNet forward (per face):                    │
   Model Forward       │    for organ in {LE, RE, N, M, J}:                  │
                       │      OrganGAT:                                      │
                       │        Linear(6→64)                                 │
                       │        GATConv(64→64, heads=4) [residual, ELU]      │
                       │        GlobalAttentionPooling → emb (256-dim)       │
                       │        MLP(256→32→1) → 4·sigmoid+1 → local_score    │
                       │                                                     │
                       │    organ_embeds = stack 5 organ → (B, 5, 256)       │
                       │    Cross-organ MultiheadAttention(heads=4)          │
                       │    → mean-pool → global_mlp(256→64→1)               │
                       │    → 4·sigmoid+1 → global_score                     │
                       │  [model.py: OrganGAT, FaceRankNet]                  │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 9              │  L_reg  = ½(wMSE(global_A, rate_A) + wMSE(B, rate_B))│
   Loss & GradNorm     │          wMSE: weighted MSE via LDS weights          │
                       │  L_rank = log1p(exp(s_B − s_A))  ∀ organ confident  │
                       │  L_div  = −Var(local_scores)                        │
                       │                                                     │
                       │  rank_scale curriculum:                             │
                       │     ep ≤ 10 → 0     (freeze)                        │
                       │     ep ≤ 20 → linear ramp 0 → 1                     │
                       │     ep > 20 → 1.0   ← reset_L0() at epoch 21        │
                       │                                                     │
                       │  GradNorm balances [L_reg, rank_scale·L_rank]       │
                       │  total = GradNorm(...) + 0.01 · L_div               │
                       │  → clip_grad_norm(5.0) → optimizer.step()           │
                       │  → scheduler.step(PCC) [ReduceLROnPlateau           │
                       │    mode=max, factor=0.5, patience=3]                │
                       │  [loss.py, train.py]                                │
                       └────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────┐
   STEP 10             │  PCC, MAE, DPD on test set                          │
   Validation          │  Spearman ρ(pseudo, pred local) per organ           │
                       │  Checkpoint saved if PCC improves                   │
                       │  [evaluate.py: run_full_evaluation,                 │
                       │   validate_local_scores]                            │
                       └─────────────────────────────────────────────────────┘
```

## 1.2. Komponen Inti yang Aktif

| Komponen | Implementasi | Fungsi |
|----------|--------------|--------|
| Landmark extraction | MediaPipe Face Mesh (468 nodes) | Geometry-only, eliminasi tekstur/warna |
| Normalisasi | Centroid + inter-ocular distance | Scale-invariance |
| Sub-graph | 5 organ: left_eye, right_eye, nose, mouth, jawline; fully-connected + self-loops | Anatomical decomposition |
| Node feature | 6-dim: `(x, y, z, Δx, Δy, Δz)`; Δ = coord − population_mean | Averageness signal di input level |
| **Pseudo-label** | **Beauty-axis projection** + percentile rank, per ethnicity | Direction-based attractiveness (Said & Todorov) |
| Pair filter | H2 rating-consistent + margin 0.3 | Konsistensi gradien & noise floor |
| Class-balance | WeightedRandomSampler bucket `(2, 3, 4)` sqrt smoothing | Cure regression-to-mean |
| Augmentation | Jitter σ=0.003 (aktif); flip tersedia tapi tidak diaktifkan | Diversity |
| OrganGAT | Linear(6→64) → GATConv heads=4 → AttnPool → MLP(256→32→1) → 4·sigmoid+1 | Per-organ scoring |
| Fusion | Cross-organ `MultiheadAttention(embed=256, heads=4)` → mean → MLP → 4·sigmoid+1 | Global score dgn inter-organ structure |
| Loss | GradNorm(L_reg, L_rank) + 0.01·L_div | α=0.5; L_div fixed karena negatif |
| Curriculum | L_rank freeze 10 ep + warmup 10 ep + reset L0 | Stabilisasi GradNorm |
| **LDS** | **Inverse-KDE weight pada L_reg** | Up-weight tail samples (Yang et al. 2021 ICML) |
| **Hard Pair Sampling** | **Stratified bucket-distance sampling di PairDataset** | Jelek↔Cantik pair jauh lebih sering muncul |
| **LR Scheduler** | **ReduceLROnPlateau(mode=max, factor=0.5, patience=3)** | LR terpotong ½ saat PCC plateau |
| Training infra | Modal A100; persistent volumes; STOP file | Migrasi dari Colab T4 |

## 1.3. Hipotesis & Refinement yang Tertanam di Kode

- **Averageness Hypothesis (Langlois & Roggman 1990)** — bukan dipakai mentah, melainkan diperluas menjadi **direction-based attractiveness**:
  - **Said & Todorov (2011)**: attractiveness adalah arah di face space, bukan proximity ke prototype.
  - **DeBruine & Jones (2007)**: caricature beautiful faces makin jauh dari mean tetap makin cantik — mereka di sepanjang **beauty axis**, bukan di titik prototype.
- **Beauty axis** = `beauty_prototype − population_mean` (per organ, per ethnicity).
- **H1 — per-ethnicity**: dua reference (mean + prototype) dihitung untuk Asian & Caucasian terpisah.
- **H2 — holistic-consistent pair**: pair masuk training hanya kalau rating(A) > rating(B).
- **H3 — confidence margin**: organ_mask True hanya jika `pseudo(A) − pseudo(B) > 0.3` (di atas noise floor).

---

# Bagian 2 — Evolusi Codebase

## 2.1. Riwayat Commit

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
| `52464be` | Fix: Train Tail and test pseudo-labels | 2026-06-01 |
| `ac45bd7` | add: lr scheduler | 2026-06-01 |

## 2.2. File yang TIDAK Berubah Sejak Awal

- [model/organ_indices.py](model/organ_indices.py) — indeks landmark per organ
- [model/evaluate.py](model/evaluate.py) — definisi PCC / MAE / DPD / Spearman validation

Sisanya (config, preprocessing, dataset, pseudo_labels, model, loss, train) **berubah signifikan**, plus folder `modal/` ditambahkan.

## 2.3. Perubahan per Modul

### a) [model/config.py](model/config.py)

| Param | Awal | Sekarang | Alasan |
|-------|------|----------|--------|
| `NODE_FEAT_DIM` | 3 | 6 | Deviasi `coord − population_mean` ditambah ke input |
| `NUM_TASKS` | 3 | 2 | L_div keluar dari GradNorm (negatif) |
| `LDIV_WEIGHT` | — | 0.01 | Fixed weight di luar GradNorm |
| `PAIRS_PER_SAMPLE` | 1 | 3 | Lebih banyak negatif per anchor |
| `GRADNORM_ALPHA` | 1.5 | 0.5 | Restoring force lebih lembut |
| `CROSS_ORGAN_HEADS` | — | 4 | Heads untuk cross-organ attention |
| `RANK_FREEZE_EPOCHS` / `RANK_WARMUP_EPOCHS` | — | 10 / 10 | Curriculum untuk L_rank |
| `RANK_PSEUDO_MARGIN` | — | 0.3 | Confidence margin (H3) |
| `USE_WEIGHTED_PAIR_SAMPLER` | — | True | Class-balance bucket rating |
| `PAIR_SAMPLER_BUCKET_EDGES` / `_SMOOTHING` | — | (2,3,4) / "sqrt" | 4 bucket + boost ~3.6× minoritas |
| `AUGMENT_JITTER` / `JITTER_STD` | — | True / 0.003 | Per-sample noise |
| `RESUME_FROM_CHECKPOINT` | — | True | Default resume |
| **Sprint 2 — Tail fixes** | | | |
| `USE_INVERSE_FREQ_L_REG` | — | True | LDS: bobot invers-densitas pada L_reg |
| `LREG_LDS_BANDWIDTH` | — | 0.3 | Bandwidth KDE untuk estimasi densitas rating |
| `LREG_WEIGHT_FLOOR` | — | 0.2 | Min weight agar tail tidak overflow |
| `USE_HARD_PAIR_SAMPLING` | — | True | HPS: stratified by bucket distance |
| `USE_WITHIN_BUCKET_MIXUP` | — | False | MixUp training-time (off; paradigma B lebih baik) |
| `MIXUP_PROB` / `MIXUP_ALPHA` | — | 0.5 / 0.4 | Probabilitas & parameter Beta distribution |
| `MIXUP_TAIL_BUCKETS_ONLY` / `MIXUP_BUCKET_EDGES` | — | True / (2,3,4) | Hanya Jelek+Cantik bucket |

### b) [model/preprocessing.py](model/preprocessing.py)

- `build_subgraph` & `build_all_subgraphs` menerima `avg_face` opsional. Bila diberikan → fitur node 6-dim (`coords + deviation`).
- Fungsi baru `build_all_subgraphs_flipped`: X-axis dibalik + label `left_eye` ↔ `right_eye` ditukar (anatomis-correct flip augmentation).

### c) [model/dataset.py](model/dataset.py)

- `FaceDataset`: parameter `avg_face`, `augment_flip`, `augment_jitter`, `jitter_std`. Jitter pada `__getitem__` setiap kali sample diambil.
- `PairDataset._build_pairs`: filter **H2** (`rating_a > rating_b`) + **margin H3** (`(pseudo_a − pseudo_b) > RANK_PSEUDO_MARGIN`) yang menentukan `organ_mask`.
- `make_weighted_pair_loader`: WeightedRandomSampler bucket rating untuk class balance.

**Sprint 2 additions:**

- `FaceDataset` — parameter baru `compute_lds_weights` dan `mixup_within_bucket`.
  - `_build_lds_weights(ratings)` — KDE via `scipy.stats.gaussian_kde`, kembalikan bobot invers-densitas dinormalisasi mean=1.
  - `__getitem__` kini mengembalikan `rating_weight` (scalar) untuk dikirim ke `l_reg`.
  - Within-bucket MixUp di `__getitem__`: bila aktif, campurkan coords + rating + pseudo secara linear dengan partner dari bucket yang sama (`Beta(0.4, 0.4)`). Default OFF.
- `PairDataset` — parameter baru `hard_pair_sampling`.
  - `_sample_candidates(a_idx, n, k)`: tanpa HPS = random; dengan HPS = stratified by `abs(bucket_b − bucket_a) + 1` sebagai weight → pasangan ekstrem (Jelek↔Cantik) muncul jauh lebih sering dari ~0.5%.
- `collate_faces` — stack `rating_weights` ke dalam batch dict.

### d) [model/pseudo_labels.py](model/pseudo_labels.py)

- `compute_universal_average_face` — population mean global / per-ethnicity (dipakai sebagai μ untuk beauty axis).
- `compute_beauty_prototype` — mean top-k% (default 30%) wajah tertinggi (endpoint beauty axis).
- `compute_ethnicity_avg_faces` — populasi mean **dan** beauty prototype dihitung per kelompok etnis (H1).
- **`compute_beauty_axis`**, **`project_organ_onto_axis`** — operator inti pseudo-label: konstruksi axis dan scalar projection per organ.
- **`compute_all_pseudo_labels_beauty_axis`** — fungsi yang **dipanggil di notebook Cell 5** untuk menghasilkan `pseudo_labels.pkl`. Memakai `population_mean_map` + `beauty_prototype_map` per etnis + `ethnicity_map`. Output: percentile rank atas proyeksi → score ∈ [1, 5].
- `compute_all_pseudo_labels` (RMSE + percentile) — versi lama, masih ada di codebase tapi **tidak dipakai** oleh notebook saat ini.
- `validate_pseudo_label_quality` — Spearman ρ diagnostic.

**Sprint 2 — 7 metode pseudo-label baru (untuk benchmark/ablasi):**

| Nama metode | Fungsi utama | Catatan |
|-------------|-------------|---------|
| `axis_eth` | Beauty-axis per ethnicity (sudah ada) | Baseline saat ini |
| `axis_eth_gen` | `compute_subgroup_avg_faces` — beauty axis per (eth × gender) | Prototype dari subgroup lebih homogen |
| `axis_eth_kmeans` | `compute_all_pseudo_labels_beauty_axis_kmeans` — K=3 sub-prototypes per eth via KMeans, ambil max-projection | Atasi feature cancellation saat prototype dirata-rata dari wajah cantik beragam bentuk |
| `axis_eth_quantile` | `compute_all_pseudo_labels_quantile` — remap distribusi pseudo ke distribusi GT rating via quantile matching | Rank order dipertahankan (Spearman ρ sama), skala disesuaikan ke Gaussian-ish GT |
| `axis_eth_synthaug` | `compute_all_pseudo_labels_axis_synthaug` — pseudo-label-time MixUp: generate synthetic tail faces dari campuran landmark, lalu hitung pseudo-label geometric dari landmark campur tersebut (Paradigma B) | Lebih akurat dari training-time MixUp karena pseudo-label berasal dari geometri aktual |
| `symmetry` | `compute_all_pseudo_labels_symmetry` — skor bilateral simetri per organ (standalone) | Independent dari beauty axis |
| `canons` | `compute_all_pseudo_labels_canons` — deviasi dari kanon neoklasik (rasio wajah ideal) | Independent dari beauty axis |
| `axis_quantile_canons_sym` | `compute_all_pseudo_labels_blend` — equal-weight blend dari `axis_eth_quantile` + `symmetry` + `canons` | Menggabungkan tiga sinyal keindahan berbeda |

- `pseudo_label_quality_report` + `print_benchmark_leaderboard` — diagnostic lengkap: Spearman ρ, KS distance, kalibrasi per-organ; output tabel perbandingan semua metode.
- Seluruh metode baru terdokumentasi di [pseudolabel.md](pseudolabel.md).

### e) [model/model.py](model/model.py)

- `OrganGAT` dimodifikasi: forward bisa return `(score, pooled_embedding)` untuk fusion.
- `FaceRankNet`:
  - Tambah `nn.MultiheadAttention(embed_dim=256, num_heads=4)`.
  - Tambah `global_mlp(256→64→1)`.
  - `fusion_weights` dipertahankan **hanya** untuk interpretability (organ importance), tidak menentukan global_score.
  - Forward output kini menyertakan `attn_weights (B, 5, 5)` yang visualizable.

### f) [model/loss.py](model/loss.py)

- `GradNorm._get_shared_params`: anchor di `OrganGAT.mlp[-1]` (Linear 32→1) — menerima gradien dari L_reg **dan** L_rank. Sebelumnya hanya `input_proj` (cuma L_reg).
- `GradNorm.reset_L0()`: memaksa re-capture L0 setelah L_rank selesai warmup. Tanpa reset, L0 yang ter-clamp ke 1e-8 dari fase freeze merusak rasio loss seterusnya.

**Sprint 2 addition:**

- `l_reg(global_pred, global_gt, weights=None)` — parameter opsional `weights` (Tensor shape B). Bila diberikan: weighted MSE = `(w * (pred−gt)²).sum() / w.sum()`; bila None: fallback ke `F.mse_loss` biasa.

### g) [model/train.py](model/train.py)

- Load `avg_face` (=population_mean global) + diagnostic Spearman ρ sebelum training.
- Pair pool di-rebuild setiap epoch.
- WeightedRandomSampler dibangun ulang per epoch.
- `L_rank` curriculum (freeze 10 + warmup 10 + reset L0 di epoch 21).
- L_reg = rata-rata MSE(A) + MSE(B).
- L_div ditambahkan dengan bobot tetap di luar GradNorm.
- `clip_grad_norm_(max_norm=5.0)`.
- Resume from checkpoint (state model + optimizer + lambdas + best_pcc + epoch).
- File `STOP` untuk graceful halt.

**Sprint 2 additions:**

- `FaceDataset` train diinisialisasi dengan `compute_lds_weights=config.USE_INVERSE_FREQ_L_REG` dan `mixup_within_bucket=config.USE_WITHIN_BUCKET_MIXUP`. Test set tetap bersih (`compute_lds_weights=False`).
- `PairDataset` diinisialisasi dengan `hard_pair_sampling=config.USE_HARD_PAIR_SAMPLING`.
- Di training loop: `w_a`, `w_b` diambil dari `batch["rating_weights"]` bila `USE_INVERSE_FREQ_L_REG` aktif, lalu diteruskan ke `l_reg(weights=w_a/w_b)`.
- `ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)` — `scheduler.step(pcc)` dipanggil setelah setiap epoch validasi. LR terpotong ½ bila PCC tidak naik selama 3 epoch berturut-turut.

### h) [modal/](modal/)

Migrasi training dari Colab T4 ke Modal A100.

- [modal/app.py](modal/app.py) — baseline training. Volume: `frn-data` (CSV), `frn-cache` (landmarks/pseudo/avg_face), `frn-checkpoints` (best ckpt). Patch path config sebelum `import train`.
- [modal/app_dgf.py](modal/app_dgf.py) — varian DeepGeoFusion paralel (Delaunay + edge features), checkpoint terpisah `frn-checkpoints-dgf`.
- [modal/diagnostic.py](modal/diagnostic.py) — visualisasi graph topology + edge feature.
- [modal/upload_cache.py](modal/upload_cache.py) — upload CSV/cache/checkpoint dari lokal ke Modal Volumes.

GPU `A100-40GB`, timeout 6 jam, `--detach` untuk run yang survive disconnect.

---

## 2.4. Ringkasan Padat

Pipeline saat ini berbeda dari versi awal dalam **delapan dimensi**:

1. **Input** — node feature 6-dim: `coords + (coords − population_mean)`.
2. **Pseudo-label** — bukan jarak ke mean, tapi **proyeksi skalar pada beauty axis** `(prototype − mean)` + percentile rank, per ethnicity. 7 metode alternatif tersedia untuk benchmark.
3. **Pair sampling** — H2 (holistic-consistent) + H3 confidence margin τ=0.3 + resampling per epoch + class-balanced WeightedRandomSampler + **Hard Pair Sampling** (bucket-distance stratified).
4. **Arsitektur** — cross-organ MultiheadAttention menggantikan softmax fusion sederhana.
5. **Optimisasi** — GradNorm 2 task (L_div fixed), α=0.5, anchor di MLP-tail, L_rank curriculum (freeze + warmup + L0 reset), gradient clipping.
6. **Tail fix** — **LDS** (inverse-KDE weight pada L_reg) memerangi score compression di Jelek/Cantik.
7. **LR Scheduler** — `ReduceLROnPlateau(mode=max, factor=0.5, patience=3)` memecah PCC plateau.
8. **Infra** — Colab T4 → Modal A100 dengan volume persistent + checkpoint resume + STOP file.

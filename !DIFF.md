# !DIFF — Paper vs Code: FaceRankNet

Dokumen ini mencatat **semua perbedaan antara klaim di paper (`FaceRankNet_Paper.md`) dan implementasi aktual di codebase**. Digunakan sebagai referensi sebelum submission — bagian yang berbeda perlu diperbarui di paper atau justified sebagai improvement yang disengaja.

---

## DIFF 1 — Node Feature Dimension

| | Detail |
|--|--------|
| **Paper** (§III.B, §III.C.1) | `F ∈ ℝ^{468×3}` — hanya koordinat `(x, y, z)`. "Each node's normalized coordinates are projected..." |
| **Code** ([config.py:46](model/config.py#L46), [preprocessing.py:237](model/preprocessing.py#L237)) | `NODE_FEAT_DIM = 6` — `(x, y, z, Δx, Δy, Δz)`. Deviasi dari reference face langsung dimasukkan ke node feature. |
| **Impact** | `Linear(3→64)` di paper vs `Linear(6→64)` di code. Model yang dilatih tidak kompatibel dengan deskripsi paper. |
| **Aksi** | Update paper §III.B dan §III.C.1: tambahkan deskripsi 6-dim node feature dan motivasinya (averageness signal eksplisit di input level). |

---

## DIFF 2 — Formula Pseudo-Label: MSE + Linear vs RMSE + Percentile

| | Detail |
|--|--------|
| **Paper** (§III.D.1) | Menggunakan **MSE** (squared): $MSE = \frac{1}{N}\sum\|p_i - \mu_i\|^2$. Normalisasi linear: $\hat{y}^{psc} = 5 - 4 \cdot \frac{MSE}{\max(MSE)}$ |
| **Code** ([pseudo_labels.py:104](model/pseudo_labels.py#L104), [pseudo_labels.py:368](model/pseudo_labels.py#L368)) | Menggunakan **RMSE** (square root). Normalisasi **percentile rank**: `rank = bisect_left(sorted_mse, mse) / n` → `score = clip(5 − 4·rank, 1, 5)` |
| **Mengapa berbeda** | Linear normalization terhadap `max_MSE` menyebabkan score compression (mean 3.7–4.5, hampir semua wajah terlihat cantik). Percentile rank menghasilkan distribusi uniform di [1,5], meningkatkan discriminative power. RMSE dipakai untuk konsistensi satuan antar organ. |
| **Aksi** | Update paper §III.D.1: (1) ganti MSE → RMSE, (2) ganti formula linear → percentile rank, (3) tambahkan motivasi: "menghindari efek kompresi akibat outlier tunggal." |

---

## DIFF 3 — Referensi Wajah: Population Average vs Beauty Prototype

| | Detail |
|--|--------|
| **Paper** (§III.D.1) | "Universal Average Face is constructed by computing the coordinate-wise mean of **all** 468 normalized landmarks across the **entire** training set." |
| **Code** ([pseudo_labels.py:147](model/pseudo_labels.py#L147), [pseudo_labels.py:250](model/pseudo_labels.py#L250)) | Menggunakan **Beauty Prototype**: mean dari **top-30% wajah dengan holistic rating tertinggi**, dihitung **per etnicity** (Asian & Caucasian terpisah) lewat `compute_ethnicity_avg_faces` + `compute_beauty_prototype`. |
| **Motivasi** | Population average mencampur wajah cantik + jelek → referensi tidak representatif. Diagnosis menunjukkan Spearman ρ = −0.13 dengan population average. Beauty prototype mengisolasi "attractive subspace" sehingga hypothesis averageness tetap valid. |
| **Aksi** | Update paper §III.D.1 dan §III.C (Weakly Supervised Learning): deskripsikan beauty prototype sebagai "Refined Universal Average Face" dari top-k% training faces per ethnicity. Tambahkan ke §I (Introduction) sebagai bagian dari H1 refinement. |

---

## DIFF 4 — H2: Holistic Consistency Filter + Confidence Margin

| | Detail |
|--|--------|
| **Paper** (§III.D.2) | Pair (A, B) dipilih berdasarkan **pseudo-score ordering** saja: jika `pseudo(A) > pseudo(B)` untuk suatu organ, model dilatih untuk memprediksi `local_score(A) > local_score(B)`. |
| **Code** ([dataset.py:204](model/dataset.py#L204), [dataset.py:234](model/dataset.py#L234)) | Dua filter tambahan: (1) **H2**: skip pair jika `rating_a <= rating_b`. (2) **Confidence margin**: `organ_mask = (pseudo_a − pseudo_b) > RANK_PSEUDO_MARGIN` (default 0.3). Organ dengan gap di bawah noise floor di-mask 0. |
| **Motivasi** | H2: tanpa filter, L_rank bisa dilatih untuk memenangkan ranking yang bertentangan dengan L_reg → konflik gradien → λ_rank meledak. Margin: pseudo-label noisy (ρ ≈ 0.57); gap kecil sering kebalik tanda, jadi hanya gap konfiden yang dipakai. |
| **Aksi** | Tambahkan ke paper §III.D.2 sebagai "Consistency-Filtered Pair Sampling" + "Confidence Margin Filter". Sebutkan margin 0.3 dan justifikasinya. |

---

## DIFF 5 — GradNorm: 3 Tasks vs 2 Tasks (+ α berbeda)

| | Detail |
|--|--------|
| **Paper** (§III.D.3, §III.D.4) | Formula total loss: $\mathcal{L} = \lambda_1\mathcal{L}_{reg} + \lambda_2\mathcal{L}_{rank} + \lambda_3\mathcal{L}_{div}$ — mengimplikasikan GradNorm mengelola **ketiga** loss secara dinamis. Hyper-parameter $\alpha = 1.5$. |
| **Code** ([config.py:74](model/config.py#L74), [train.py:296](model/train.py#L296)) | GradNorm hanya mengelola **2 task**: `[L_reg, L_rank]`. `L_div` ditambahkan dengan **bobot tetap** `LDIV_WEIGHT = 0.01` di luar GradNorm. `GRADNORM_ALPHA = 0.5` (bukan 1.5). |
| **Mengapa berbeda** | `L_div = −Var(scores)` bernilai **negatif** — melanggar asumsi positivitas GradNorm. α=0.5 (restoring force lebih lembut) menstabilkan λ; α=1.5 menyebabkan swing besar. |
| **Aksi** | Update paper §III.D.3: pisahkan $\mathcal{L}_{div}$ dari GradNorm. Formula baru: $\mathcal{L}_{total} = \text{GradNorm}_{\alpha=0.5}(\lambda_1\mathcal{L}_{reg}, \lambda_2\mathcal{L}_{rank}) + 0.01 \cdot \mathcal{L}_{div}$. Tambahkan kalimat: "L_div dikecualikan karena bernilai negatif, bertentangan dengan asumsi loss positif GradNorm; α dikecilkan menjadi 0.5 untuk menstabilkan λ." |

---

## DIFF 6 — L_reg Menggunakan Kedua Wajah dalam Pair

| | Detail |
|--|--------|
| **Paper** (§III.D.3) | $\mathcal{L}_{reg} = \frac{1}{N}\sum(\hat{y}_{global} - y_{gt})^2$ — tidak dispesifikasikan apakah menggunakan satu atau dua wajah per pair. |
| **Code** ([train.py:276](model/train.py#L276)) | `loss_reg = (l_reg(pred_a, rating_a) + l_reg(pred_b, rating_b)) / 2` — L_reg dihitung dari **kedua** wajah A dan B per batch. |
| **Motivasi** | Menggunakan hanya face A membuang 50% ground-truth signal holistic per batch. Menggunakan A+B menggandakan sinyal regresi tanpa menambah data baru. |
| **Aksi** | Update paper §III.D.3: tambahkan keterangan bahwa L_reg dihitung dari kedua wajah dalam setiap pair untuk memaksimalkan utilitas ground-truth label. |

---

## DIFF 7 — Pair Resampling Per Epoch (Tidak Ada di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. |
| **Code** ([train.py:236](model/train.py#L236)) | `pair_ds._pairs = pair_ds._build_pairs()` dipanggil di awal **setiap epoch** — pair di-resample ulang sehingga model tidak overfit pada set pasangan tetap. |
| **Motivasi** | Pair tetap menyebabkan model menghafalkan urutan spesifik antar wajah, bukan belajar pola umum. Resampling per epoch setara dengan data augmentation untuk ranking. |
| **Aksi** | Tambahkan ke paper §III.D.2: "Pairs are resampled at the beginning of each epoch to prevent overfitting to a fixed pair ordering." |

---

## DIFF 8 — GAT Multi-Head (Tidak Dispesifikasikan di Paper)

| | Detail |
|--|--------|
| **Paper** (§III.C.2) | Formula GAT standar ditampilkan — tidak menyebutkan jumlah head. Terkesan single-head. |
| **Code** ([config.py:52](model/config.py#L52), [model.py:90](model/model.py#L90)) | `GAT_NUM_HEADS = 4` — 4 parallel attention heads. Output shape `(N, 4, 64)` → flatten → `(N, 256)`. |
| **Aksi** | Update paper §III.C.2: tambahkan "multi-head attention with K=4 heads" dan modifikasi formula menjadi multi-head: $h'_i = \|_{k=1}^{K} \sigma\left(\sum_j \alpha_{ij}^k W^k h_j\right)$ |

---

## DIFF 9 — Sub-Graph Connectivity (Tidak Dispesifikasikan di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak menyebutkan struktur edge dalam organ sub-graph. |
| **Code** ([preprocessing.py:225](model/preprocessing.py#L225)) | Sub-graph dibangun sebagai **fully connected** (semua node dalam satu organ terhubung ke semua node lain) + **self-loops** via `dgl.add_self_loop`. |
| **Aksi** | Tambahkan ke paper §III.B.3: "Each organ sub-graph is constructed as a fully-connected graph with self-loops, allowing every landmark to attend to all others within the same anatomical region." |

---

## DIFF 10 — Local Score Validity: Definisi Validasi

| | Detail |
|--|--------|
| **Paper** (§III.E) | "Verifying that facial components geometrically **closer to the Universal Average Face** receive proportionally **higher** aesthetic scores." |
| **Code** ([evaluate.py:116](model/evaluate.py#L116)) | Menghitung **Spearman ρ antara pseudo_scores dan predicted local_scores**. Validasi = semua 5 organ punya ρ > 0. Tidak langsung mengukur jarak ke avg face — melainkan konsistensi dengan pseudo-label. |
| **Implikasi setelah beauty prototype** | Setelah DIFF 3 (beauty prototype), pseudo-scores: rendah RMSE dari *beauty prototype* → score tinggi. |
| **Aksi** | Update paper §III.E: ganti "Universal Average Face" → "Beauty Prototype". Klarifikasi bahwa validasi mengukur Spearman ρ antara predicted organ scores dan pseudo-scores berbasis beauty prototype. |

---

## DIFF 11 — Fusion: Softmax-Weighted vs Cross-Organ MultiheadAttention

| | Detail |
|--|--------|
| **Paper** (§III.C.3) | "Global score = Σ_k softmax(w_k) · local_score_k" — fusion linear lewat learnable scalar weights di atas **scalar local scores**. |
| **Code** ([model.py:191](model/model.py#L191), [model.py:236](model/model.py#L236)) | Fusion lewat `nn.MultiheadAttention(embed_dim=256, num_heads=4)` di atas **organ embeddings** (B, 5, 256). Mean-pool attended embeddings → `global_mlp(256→64→1)` → `4·sigmoid+1`. `fusion_weights` masih ada tetapi **hanya untuk interpretability** (organ importance), tidak masuk perhitungan global_score. |
| **Motivasi** | Softmax atas skalar membuang struktur fine-grained antar organ. Cross-organ attention memungkinkan model menangkap proporsi antar bagian wajah (golden ratio, simetri eye-nose-mouth, dll). Memberikan attention map (B, 5, 5) yang bisa divisualisasikan. |
| **Aksi** | Update paper §III.C.3 secara substansial: ganti formula softmax → cross-organ MultiheadAttention. Definisikan embed_dim=256 (= GAT_HIDDEN_DIM × GAT_NUM_HEADS), 4 heads, mean-pool + MLP head. Posisikan ini sebagai kontribusi arsitektur (label B2 di internal notes). |

---

## DIFF 12 — L_rank Warmup (Freeze + Linear Ramp)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan — L_rank diasumsikan aktif dari epoch 1. |
| **Code** ([train.py:280-289](model/train.py#L280), [config.py:120-125](model/config.py#L120)) | L_rank dikalikan `rank_scale` yang naik bertahap: epoch 1..10 → 0 (freeze), epoch 11..20 → linear ramp 0→1, epoch 21+ → 1. Setelah warmup selesai, [`gradnorm.reset_L0()`](model/loss.py#L181) memaksa re-capture L0 (kalau tidak, L0 yang ter-clamp ke 1e-8 saat L_rank=0 merusak balancing seterusnya). |
| **Motivasi** | Pengaktifan tiba-tiba L_rank shock GradNorm: gradient ratio meledak, λ berayun. Freeze memberi L_reg waktu menstabilkan regresi baseline; warmup menambahkan L_rank secara halus. |
| **Aksi** | Tambahkan ke paper §III.D.4 sub-bagian "Training Schedule": jelaskan freeze (10 ep) + linear warmup (10 ep) + L0 reset. Argumen: tanpa skema ini, λ_rank terbang ke 5+ di epoch 1–3 dan training collapse. |

---

## DIFF 13 — Class-Balanced Pair Sampling (Tidak Ada di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. Asumsi sampling uniform. |
| **Code** ([dataset.py:318](model/dataset.py#L318), [train.py:240](model/train.py#L240)) | `make_weighted_pair_loader` pakai `WeightedRandomSampler` dengan bucket rating `(2.0, 3.0, 4.0)` (4 kelas: Jelek <2, 2–3, 3–4, Cantik >4). Bobot `1/√count` (sqrt smoothing). Jelek (~4.7%) dan Cantik (~11%) di-boost ~3.6×. |
| **Motivasi** | SCUT-FBP5500 heavy-imbalanced di sekitar mean (rating 2.5–3.5). Tanpa rebalance, model jatuh ke "regression-to-mean" — selalu prediksi ~3. Bucket-weighted sampling memastikan minoritas ekstrem muncul cukup sering. |
| **Aksi** | Tambahkan ke paper §III.D.2 atau §IV (Implementation): "Pair anchors are resampled per epoch with bucket-weighted probability ∝ 1/√count over the 4 rating buckets {<2, 2–3, 3–4, >4}, mitigating regression-to-mean on the imbalanced SCUT-FBP5500 distribution." |

---

## DIFF 14 — Data Augmentation: Jitter + Horizontal Flip

| | Detail |
|--|--------|
| **Paper** | Tidak menyebutkan augmentation. |
| **Code** ([dataset.py:122](model/dataset.py#L122), [preprocessing.py:278](model/preprocessing.py#L278), [config.py:136-137](model/config.py#L136)) | (1) **Jitter**: `coords + N(0, σ²)` dengan σ=0.003 (~0.3% pasca inter-ocular normalization), diterapkan setiap `__getitem__`. (2) **Flip**: opsi `augment_flip` di `FaceDataset` menduplikasi dataset dengan wajah ter-mirror X-axis, label `left_eye` ↔ `right_eye` ditukar. |
| **Motivasi** | Jitter mencegah overfit pada minoritas yang di-resample berulang oleh WeightedSampler (set unik <200 face). Flip mengeksploitasi simetri bilateral wajah. |
| **Aksi** | Tambahkan ke paper §IV: "Landmark-space augmentations: Gaussian jitter (σ=0.003) per sample and optional horizontal mirroring with anatomically-correct left/right swap." |

---

## DIFF 15 — Gradient Clipping (Tidak Ada di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. |
| **Code** ([train.py:301](model/train.py#L301)) | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` setelah `total_loss.backward()`. |
| **Motivasi** | L_rank softmax-margin (`log1p(exp(s_B − s_A))`) bisa meledak ketika selisih besar — clip mencegah update destruktif. |
| **Aksi** | Tambahkan kalimat singkat di §IV (Implementation): "Gradient norm clipped at 5.0 to prevent loss spikes from extreme ranking margins." |

---

## DIFF 16 — Beauty Axis Projection (Implementasi Alternatif, Belum di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. |
| **Code** ([pseudo_labels.py:188](model/pseudo_labels.py#L188), [pseudo_labels.py:386](model/pseudo_labels.py#L386)) | Fungsi `compute_beauty_axis`, `project_organ_onto_axis`, `compute_all_pseudo_labels_beauty_axis`: pseudo-label berdasarkan proyeksi skalar `(face − population_mean) · (beauty_prototype − population_mean)`. Bukan jarak ke prototype, melainkan arah sepanjang "beauty axis". |
| **Motivasi** | Said & Todorov (2011), DeBruine & Jones (2007): attractiveness adalah **arah** di face space, bukan proximity ke titik tunggal. Caricature beautiful faces lebih jauh dari mean tapi lebih cantik. |
| **Aksi** | Bila digunakan untuk ablation, paper §III.D.1 perlu menambahkan formulasi alternatif sebagai eksperimen pembanding. Bila tidak dipakai untuk experimen utama, biarkan sebagai kode referensi tanpa update paper. |

---

## DIFF 17 — Modal A100 Deployment (Infra, Belum di Paper)

| | Detail |
|--|--------|
| **Paper** (§IV / Implementation) | Disebut: "Implemented in PyTorch + DGL. Training on Google Colab T4." |
| **Code** ([modal/app.py](modal/app.py), [modal/app_dgf.py](modal/app_dgf.py), [modal/upload_cache.py](modal/upload_cache.py)) | Training migrasi ke **Modal A100-40GB**. Tiga volume persistent: `frn-data` (CSV), `frn-cache` (landmarks/pseudo/avg_face), `frn-checkpoints` (best checkpoint). `app_dgf.py` menyiapkan jalur paralel untuk varian DeepGeoFusion (`model_dgf/`) dengan checkpoint terpisah. |
| **Aksi** | Update paper §IV: ganti "T4" → "Modal A100-40GB"; sebutkan resume-from-checkpoint mechanism. Bila DGF jadi bagian paper, tambahkan deskripsi setup paralel. |

---

## Ringkasan Aksi Paper

| DIFF | Bagian Paper | Jenis Perubahan |
|------|-------------|-----------------|
| 1 | §III.B, §III.C.1 | Update formula & motivasi 6-dim node feature |
| 2 | §III.D.1 | Ganti MSE→RMSE, linear→percentile, tambah motivasi |
| 3 | §I, §III.C, §III.D.1 | Tambah beauty prototype sebagai kontribusi H1 |
| 4 | §III.D.2 | Tambah H2 consistency filter + confidence margin |
| 5 | §III.D.3, §III.D.4 | Pisahkan L_div dari GradNorm; α 1.5 → 0.5 |
| 6 | §III.D.3 | Klarifikasi L_reg menggunakan kedua wajah A+B |
| 7 | §III.D.2 | Tambahkan pair resampling per epoch |
| 8 | §III.C.2 | Spesifikasikan K=4 multi-head GAT + formula update |
| 9 | §III.B.3 | Tambahkan fully-connected + self-loop sub-graph |
| 10 | §III.E | Update referensi: avg face → beauty prototype |
| 11 | §III.C.3 | Ganti softmax fusion → cross-organ MultiheadAttention (B2) |
| 12 | §III.D.4 | Tambah L_rank freeze + warmup + L0 reset |
| 13 | §III.D.2 / §IV | Tambah WeightedRandomSampler untuk rating bucket |
| 14 | §IV | Tambah jitter (σ=0.003) + horizontal flip augmentation |
| 15 | §IV | Tambah gradient clipping (max_norm=5.0) |
| 16 | §III.D.1 (opsional) | Beauty axis projection sebagai ablation alternatif |
| 17 | §IV | Migrasi Colab T4 → Modal A100 |

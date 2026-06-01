# !DIFF — Paper vs Code: FaceRankNet

Dokumen ini mencatat **perbedaan antara klaim di paper (`FaceRankNet_Paper.md`) dan implementasi aktual di codebase**. Tiap entri menunjuk klausa paper yang spesifik + lokasi kode terkait. Digunakan sebagai checklist revisi paper sebelum submission.

> Diff yang sudah tidak relevan (sudah disinkronkan, atau hanya kode eksperimen di luar pipeline utama, atau hal di luar klaim paper) telah dihapus dari versi sebelumnya.

---

## DIFF 1 — Node Feature Dimension (3 → 6)

| | Detail |
|--|--------|
| **Paper** §III.B.2, §III.C.1 | "Each face is represented as a matrix $F \in \mathbb{R}^{468 \times 3}$." dan "Each node's normalized coordinates are projected into a higher-dimensional embedding space" — input GAT = 3-dim. |
| **Code** [config.py:46](model/config.py#L46), [preprocessing.py:237](model/preprocessing.py#L237) | `NODE_FEAT_DIM = 6` — `(x, y, z, Δx, Δy, Δz)`. Saat `avg_face` diberikan, fitur node digabung dengan deviasi terhadap reference face. `Linear(6→64)`. |
| **Aksi paper** | §III.B.2 / §III.C.1: ganti $F \in \mathbb{R}^{468 \times 3}$ menjadi $\mathbb{R}^{468 \times 6}$. Tambahkan kalimat: "Each node feature concatenates its normalized coordinates with the per-landmark deviation $\Delta = p_i - \mu_i$ from the reference face, making the Averageness signal explicit at the input level rather than requiring the GAT to infer it from absolute coordinates alone." |

---

## DIFF 2 — Pseudo-Label Formulation: Proximity-to-Mean (MSE+Linear) → Beauty-Axis Projection + Percentile Rank

| | Detail |
|--|--------|
| **Paper** §III.D.1 | $MSE_{organ} = \frac{1}{N}\|p_i - \mu_i\|^2$. Normalisasi linear: $\hat{y}^{psc} = 5 - 4 \cdot \frac{MSE}{\max(MSE)}$. Skor naik ketika face **dekat ke mean** (proximity-to-prototype). |
| **Code** [pseudo_labels.py:188](model/pseudo_labels.py#L188), [pseudo_labels.py:217](model/pseudo_labels.py#L217), [pseudo_labels.py:386](model/pseudo_labels.py#L386); [run_colab.ipynb Cell 5](model/run_colab.ipynb) | **Beauty-axis projection**, bukan RMSE/jarak. (1) `beauty_axis = beauty_prototype − population_mean`. (2) Untuk tiap organ: `proj = (coord_organ − μ_organ) · axis_organ / ‖axis_organ‖`. (3) **Percentile rank** atas proyeksi: `rank = bisect_left(sorted_proj, proj) / n` → `score = clip(1 + 4·rank, 1, 5)`. Direction: **higher projection along the beauty axis → higher score** (bukan "lower distance"). |
| **Mengapa berbeda** | Said & Todorov (2011) + DeBruine & Jones (2007): attractiveness adalah **arah** di face space, bukan proximity ke satu titik tunggal. Wajah cantik yang ter-caricature ("hyper-beautiful") secara matematis lebih jauh dari mean tapi tetap dipersepsi lebih cantik — karena posisinya **searah** dengan beauty axis, bukan dekat dengan prototype. Diagnosa awal versi MSE-to-mean: Spearman ρ negatif (sekitar −0.13). Versi beauty-axis projection: Spearman ρ ~0.57. Linear/$\max$ juga menyebabkan kompresi distribusi → percentile rank dipilih untuk distribusi uniform di [1, 5]. |
| **Aksi paper** | §III.D.1 perlu rewrite substansial. Ganti formula MSE/jarak menjadi proyeksi skalar pada vektor `beauty_prototype − population_mean`. Jelaskan: (1) konstruksi beauty axis, (2) scalar projection per organ, (3) percentile rank → [1, 5]. Sitasi Said & Todorov 2011, DeBruine & Jones 2007 untuk justifikasi direction-based daripada proximity-based attractiveness. |

---

## DIFF 3 — Reference Face: Single Population Mean → Population Mean + Beauty Prototype, Per Ethnicity

| | Detail |
|--|--------|
| **Paper** §III.D.1 | Satu reference: "Universal Average Face by computing the coordinate-wise mean of **all** 468 normalized landmarks across the **entire** training set." |
| **Code** [pseudo_labels.py:49](model/pseudo_labels.py#L49), [pseudo_labels.py:147](model/pseudo_labels.py#L147), [pseudo_labels.py:250](model/pseudo_labels.py#L250); [run_colab.ipynb Cell 5](model/run_colab.ipynb) | **Dua reference dipakai bersama** untuk membentuk beauty axis: (1) **`population_mean`** = mean semua wajah; (2) **`beauty_prototype`** = mean dari top-30% wajah dengan rating tertinggi. Keduanya dihitung **per kelompok etnis** (Asian / Caucasian terpisah) lewat `compute_ethnicity_avg_faces` — H1 refinement. |
| **Motivasi** | Population mean saja tidak memberi arah (hanya titik). Beauty prototype mendefinisikan endpoint dari beauty axis. Per-ethnicity menghilangkan bias bila kedua kelompok punya struktur wajah berbeda. |
| **Aksi paper** | §I — tambahkan kontribusi: "Beauty-axis pseudo-label generation via per-ethnicity beauty prototype + population mean pair". §III.D.1 — ganti deskripsi reference: alih-alih satu Universal Average Face, jelaskan pasangan (population_mean_ethn, beauty_prototype_ethn) yang dipakai untuk mengkonstruksi beauty axis per organ. |

---

## DIFF 4 — Pair Sampling: + H2 Holistic Consistency Filter + Confidence Margin

| | Detail |
|--|--------|
| **Paper** §III.D.2 | "Pseudo-scores $\hat{y}^{psc}_{organ}(A)$ and $\hat{y}^{psc}_{organ}(B)$ determine the ranking direction per organ" — pair sampling hanya bergantung pada urutan pseudo-score; tidak ada filter tambahan. |
| **Code** [dataset.py:204](model/dataset.py#L204), [dataset.py:234](model/dataset.py#L234) | **H2**: `if rating_a <= rating_b: continue` — hanya pair dengan urutan holistic konsisten yang dimasukkan. **Margin filter**: `organ_mask = (pseudo_a - pseudo_b) > RANK_PSEUDO_MARGIN` (=0.3); organ dengan gap di bawah noise floor di-mask 0 di dalam L_rank. |
| **Motivasi** | H2: tanpa filter, L_rank bisa memenangkan urutan yang berlawanan arah dengan L_reg → konflik gradien, λ_rank meledak. Margin: pseudo-label noisy (ρ≈0.57 vs holistic); gap kecil sering kebalik tanda, jadi hanya pair konfiden yang dipakai. |
| **Aksi paper** | §III.D.2: tambahkan sub-bagian "Consistency-Filtered Pair Sampling": (1) filter pair jika `rating(A) ≤ rating(B)`. (2) Per-organ confidence margin τ=0.3 pada gap pseudo-score. Sertakan motivasi konsistensi gradien dan noise floor. |

---

## DIFF 5 — Hybrid Loss: GradNorm Mengelola 2 Task, L_div Fixed Weight

| | Detail |
|--|--------|
| **Paper** §III.D.3 | $\mathcal{L}_{total} = \lambda_1\mathcal{L}_{reg} + \lambda_2\mathcal{L}_{rank} + \lambda_3\mathcal{L}_{div}$ — semua tiga loss memiliki λ. |
| **Paper** §III.D.4 | "GradNorm dynamically recalibrates the loss weights ($\lambda_i$) at each backward pass" — mengimplikasikan GradNorm meng-update **semua** $\lambda_i$. |
| **Code** [config.py:75](model/config.py#L75), [train.py:290](model/train.py#L290), [train.py:296-297](model/train.py#L296) | GradNorm hanya mengelola `[L_reg, L_rank]` (`NUM_TASKS = 2`). L_div ditambahkan **fixed**: `total_loss += LDIV_WEIGHT * loss_div` dengan `LDIV_WEIGHT = 0.01`. |
| **Mengapa berbeda** | $L_{div} = -\text{Var}$ bernilai **negatif**. GradNorm mengasumsikan loss positif: $r_i = L_i / L_0$ tidak terdefinisi konsisten untuk loss bertanda. |
| **Aksi paper** | §III.D.3 / §III.D.4: ubah formula menjadi $\mathcal{L}_{total} = \text{GradNorm}(\lambda_1\mathcal{L}_{reg}, \lambda_2\mathcal{L}_{rank}) + \lambda_3 \mathcal{L}_{div}$, dengan $\lambda_3 = 0.01$ (fixed). Tambahkan kalimat: "$\mathcal{L}_{div}$ is excluded from GradNorm because its negative magnitude violates GradNorm's positive-loss assumption." |

---

## DIFF 6 — L_reg Memanfaatkan Kedua Wajah dalam Pair

| | Detail |
|--|--------|
| **Paper** §III.D.3 | $\mathcal{L}_{reg} = \frac{1}{N}\sum(\hat{y}_{global} - y_{gt})^2$ — tidak dispesifikasikan apakah dari satu atau dua wajah per pair. Naturally dibaca "satu prediksi vs satu label". |
| **Code** [train.py:276](model/train.py#L276) | `loss_reg = (l_reg(pred_a, rating_a) + l_reg(pred_b, rating_b)) / 2` — keduanya dihitung per batch. |
| **Motivasi** | Memakai hanya face A membuang 50% ground-truth signal yang sudah tersedia di setiap batch ranking. |
| **Aksi paper** | §III.D.3: klarifikasi formula menjadi $\mathcal{L}_{reg} = \frac{1}{2N}\sum_{x \in \{A,B\}}(\hat{y}_{global}(x) - y_{gt}(x))^2$ — menggunakan kedua anggota pair. |

---

## DIFF 7 — Pair Resampling per Epoch (Tidak Disebutkan di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. Asumsi default: pair pool tetap selama training. |
| **Code** [train.py:236](model/train.py#L236) | `pair_ds._pairs = pair_ds._build_pairs()` dipanggil di awal setiap epoch — pair pool dibangun ulang. |
| **Motivasi** | Pair tetap → model menghafal urutan spesifik antar wajah, bukan pola umum. Resampling = data augmentation untuk task ranking. |
| **Aksi paper** | §III.D.2: tambahkan kalimat: "Training pairs are re-sampled at the beginning of each epoch to prevent overfitting to a fixed pair set and improve generalization." |

---

## DIFF 8 — GAT Multi-Head Tidak Spesifik di Paper

| | Detail |
|--|--------|
| **Paper** §III.C.2 | Formula GAT tunggal: $h'_i = \sigma(\sum_j \alpha_{ij} W h_j)$ — tidak ada notasi multi-head ($K$, $\|$, dst). |
| **Code** [config.py:52](model/config.py#L52), [model.py:90](model/model.py#L90) | `GAT_NUM_HEADS = 4`. Output `(N, 4, 64)` → flatten → `(N, 256)`. |
| **Aksi paper** | §III.C.2: ganti formula menjadi multi-head dengan concatenation: $h'_i = \big\|_{k=1}^{K=4} \sigma(\sum_j \alpha_{ij}^k W^k h_j)$, dan tambahkan dimensi output $4 \times 64 = 256$. |

---

## DIFF 9 — Sub-Graph Connectivity Tidak Dispesifikasikan di Paper

| | Detail |
|--|--------|
| **Paper** §III.B.3 | Hanya disebut "partitioned into five anatomically defined sub-graphs" — tidak ada deskripsi struktur edge. |
| **Code** [preprocessing.py:225-232](model/preprocessing.py#L225) | Setiap sub-graph dibangun **fully-connected** (semua ordered pair antar node organ) lalu `dgl.add_self_loop` → setiap node juga punya self-edge. |
| **Aksi paper** | §III.B.3: tambahkan kalimat: "Each organ sub-graph is constructed as a fully-connected directed graph with self-loops, allowing every landmark within an anatomical region to attend to all others in a single GAT layer." |

---

## DIFF 10 — Global Score Fusion: Softmax-Weighted Sum → Cross-Organ MultiheadAttention

| | Detail |
|--|--------|
| **Paper** §III.C.5 | $\hat{y}_{global} = \sum_i \text{softmax}(w_i) \cdot \hat{y}_i$ — fusion linear dari **lima skor skalar** dengan softmax weight. |
| **Code** [model.py:191-204](model/model.py#L191), [model.py:236-245](model/model.py#L236) | (1) Setiap OrganGAT mengembalikan `(score, pooled_embedding ∈ ℝ^256)`. (2) Embedding di-stack `(B, 5, 256)` dan dimasukkan ke `nn.MultiheadAttention(embed_dim=256, num_heads=4)`. (3) Mean-pool attended embeddings → `global_mlp(256→64→1)` → `4·sigmoid(x)+1`. `fusion_weights` masih ada **hanya** untuk interpretability (organ importance), bukan menentukan global_score. |
| **Motivasi** | Softmax atas skor skalar membuang struktur fine-grained di embedding organ. Cross-organ attention memungkinkan global score memperhatikan proporsi antar organ (golden ratio, simetri, dll) di level representasi 256-dim, bukan hanya scalar. Attention map `(B, 5, 5)` juga visualizable untuk eksplainabilitas. |
| **Aksi paper** | §III.C.5 perlu rewrite. Tambahkan formula MultiheadAttention pada $\{h_{organ_1}, \dots, h_{organ_5}\}$, mean-pool, MLP head ke skalar dengan scaled sigmoid. Sebut bahwa $w_i$ disimpan untuk laporan importance saja. |

---

## DIFF 11 — L_rank Warmup (Freeze + Linear Ramp + GradNorm L0 Reset)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. Diasumsikan L_rank aktif dari epoch 1. |
| **Code** [train.py:280-289](model/train.py#L280), [config.py:120-125](model/config.py#L120), [loss.py:181](model/loss.py#L181) | `rank_scale` = 0 untuk epoch 1..`RANK_FREEZE_EPOCHS` (=10); linear 0→1 untuk epoch 11..20; 1 untuk epoch ≥21. Pada awal epoch 21, `gradnorm.reset_L0()` dipanggil supaya L0_rank (yang sebelumnya ter-clamp ke 1e-8 saat L_rank=0) di-recapture. |
| **Motivasi** | Mengaktifkan L_rank tiba-tiba menyebabkan GradNorm men-shock: rasio $L_{current}/L_0$ tidak proporsional, λ_rank berayun. Freeze memberi L_reg waktu menstabilkan baseline regresi; warmup menambahkan L_rank perlahan; reset L0 menjaga rasio tetap valid. |
| **Aksi paper** | §III.D.4 (atau sub-bagian baru "Training Schedule"): jelaskan three-stage curriculum (freeze 10 epoch → linear warmup 10 epoch → full). Tambahkan kalimat tentang L0 reset agar pembaca paham mengapa GradNorm tetap stabil setelah unfreeze. |

---

## DIFF 12 — Class-Balanced Pair Sampling (WeightedRandomSampler)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. Asumsi sampling uniform. |
| **Code** [dataset.py:318](model/dataset.py#L318), [train.py:240](model/train.py#L240) | `make_weighted_pair_loader`: bucket rating dengan edges `(2.0, 3.0, 4.0)` → 4 kelas. Sampling weight per pair $\propto 1/\sqrt{count_{bucket}}$ (smoothing "sqrt"). Jelek (<2, ~4.7%) dan Cantik (>4, ~11%) dapat boost ~3.6×. |
| **Motivasi** | SCUT-FBP5500 sangat condong ke rating tengah (2.5–3.5). Tanpa rebalance, model jatuh ke regression-to-mean — prediksi konstan ~3 menghasilkan MAE rendah tetapi PCC rendah dan gagal di tail distribusi. |
| **Aksi paper** | §III.D.2 atau §IV.A: "Pair anchors are resampled per epoch with bucket-weighted probability $\propto 1/\sqrt{n_b}$ over 4 rating buckets {$<$2, 2–3, 3–4, $>$4}, mitigating regression-to-mean on the imbalanced SCUT-FBP5500 rating distribution." |

---

## DIFF 13 — Data Augmentation: Landmark Jitter + Horizontal Flip

| | Detail |
|--|--------|
| **Paper** | Tidak ada bagian augmentasi. |
| **Code** [dataset.py:122-125](model/dataset.py#L122), [preprocessing.py:278](model/preprocessing.py#L278), [config.py:136-137](model/config.py#L136) | (1) **Jitter**: `coords += N(0, σ²)` dengan σ=0.003 (~0.3% pasca inter-ocular normalization), diterapkan setiap `__getitem__` ketika `augment_jitter=True`. (2) **Flip**: `build_all_subgraphs_flipped` membalik X-axis dan menukar label left_eye ↔ right_eye agar tetap anatomis. Aktif lewat parameter `augment_flip` di `FaceDataset`. |
| **Motivasi** | Jitter mencegah overfit pada bucket minoritas (set unik <200 wajah) yang di-resample berulang oleh WeightedSampler. Flip mengeksploitasi simetri bilateral. |
| **Aksi paper** | §IV.A: tambahkan kalimat "Landmark-space augmentations are applied: Gaussian jitter ($\sigma = 0.003$) on each sample and optional horizontal mirroring with anatomically-correct left/right organ swap." |

---

## DIFF 14 — Gradient Clipping (Tidak Disebutkan di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak ada. |
| **Code** [train.py:301](model/train.py#L301) | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` setelah `total_loss.backward()`. |
| **Motivasi** | L_rank menggunakan softmax-margin (`log1p(exp(s_B − s_A))`) yang bisa meledak saat selisih besar. Clipping mencegah update destruktif. |
| **Aksi paper** | §IV.A: "Gradient norm clipped at 5.0 to prevent loss spikes from extreme ranking margins." |

---

## DIFF 15 — Validasi Local Score: Proximity ke Avg Face → Spearman ρ vs Pseudo-Score (Beauty-Axis)

| | Detail |
|--|--------|
| **Paper** §III.E | "Verifying that facial components geometrically **closer to the Universal Average Face** receive proportionally **higher** aesthetic scores." |
| **Code** [evaluate.py:116-183](model/evaluate.py#L116) | `validate_local_scores`: untuk tiap organ menghitung Spearman ρ antara `pseudo_score` dan `predicted local_score`. Test: PASS jika kelima organ ρ > 0. Tidak mengukur jarak Euclidean langsung. |
| **Implikasi setelah DIFF 2 & 3** | Karena pseudo-score sekarang berbasis **proyeksi pada beauty axis**, validasi sebenarnya menguji apakah local score model konsisten dengan "higher projection along beauty axis → higher score" — bukan "closer to average face". |
| **Aksi paper** | §III.E: ganti pernyataan validasi. Alih-alih "closer to Universal Average Face → higher score", tulis: "organ scores must be monotonically aligned with the projection of the organ onto the beauty axis $(\beta - \mu)$." Metric: Spearman ρ antara predicted organ score dan pseudo-score, PASS = ρ > 0 pada kelima organ. |

---

## Ringkasan Aksi Paper

| DIFF | Bagian Paper | Jenis Perubahan |
|------|-------------|-----------------|
| 1  | §III.B.2, §III.C.1 | Update dim node feature 3 → 6, tambah deviasi dari population_mean |
| 2  | §III.D.1 | Proximity-to-mean (MSE+linear) → **Beauty-axis projection + percentile rank** |
| 3  | §I, §III.D.1 | Dua reference (population_mean + beauty_prototype) per ethnicity untuk konstruksi beauty axis |
| 4  | §III.D.2 | H2 holistic filter + confidence margin |
| 5  | §III.D.3, §III.D.4 | Pisahkan L_div dari GradNorm; fixed λ₃ = 0.01 |
| 6  | §III.D.3 | L_reg memakai kedua wajah dalam pair |
| 7  | §III.D.2 | Pair resampling per epoch |
| 8  | §III.C.2 | Formula GAT multi-head K=4, output 256-dim |
| 9  | §III.B.3 | Fully-connected sub-graph + self-loops |
| 10 | §III.C.5 | Softmax fusion → cross-organ MultiheadAttention |
| 11 | §III.D.4 | L_rank freeze + warmup + GradNorm L0 reset |
| 12 | §III.D.2 / §IV.A | Class-balanced WeightedRandomSampler (sqrt smoothing) |
| 13 | §IV.A | Jitter (σ=0.003) + horizontal flip augmentation |
| 14 | §IV.A | Gradient clipping (max_norm=5.0) |
| 15 | §III.E | Validitas via Spearman ρ vs pseudo-score (bukan jarak literal) |

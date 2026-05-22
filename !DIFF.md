# !DIFF — Paper vs Code: FaceRankNet

Dokumen ini mencatat **semua perbedaan antara klaim di paper (`FaceRankNet_Paper.md`) dan implementasi aktual di codebase**. Digunakan sebagai referensi sebelum submission — bagian yang berbeda perlu diperbarui di paper atau justified sebagai improvement yang disengaja.

---

## DIFF 1 — Node Feature Dimension

| | Detail |
|--|--------|
| **Paper** (§III.B, §III.C.1) | `F ∈ ℝ^{468×3}` — hanya koordinat `(x, y, z)`. "Each node's normalized coordinates are projected..." |
| **Code** ([config.py:46](model/config.py#L46), [model.py:84](model/model.py#L84)) | `NODE_FEAT_DIM = 6` — `(x, y, z, Δx, Δy, Δz)`. Deviasi dari avg face langsung dimasukkan ke node feature. |
| **Impact** | `Linear(3→64)` di paper vs `Linear(6→64)` di code. Model yang dilatih tidak kompatibel dengan deskripsi paper. |
| **Aksi** | Update paper §III.B dan §III.C.1: tambahkan deskripsi 6-dim node feature dan motivasinya (averageness signal eksplisit di input level). |

---

## DIFF 2 — Formula Pseudo-Label: MSE + Linear vs RMSE + Percentile

| | Detail |
|--|--------|
| **Paper** (§III.D.1) | Menggunakan **MSE** (squared): $MSE = \frac{1}{N}\sum\|p_i - \mu_i\|^2$. Normalisasi linear: $\hat{y}^{psc} = 5 - 4 \cdot \frac{MSE}{\max(MSE)}$ |
| **Code** ([pseudo_labels.py:99](model/pseudo_labels.py#L99), [pseudo_labels.py:247](model/pseudo_labels.py#L247)) | Menggunakan **RMSE** (square root). Normalisasi **percentile rank**: `rank = bisect(sorted_mse, mse) / n` → `score = 5 - 4 * rank` |
| **Mengapa berbeda** | Linear normalization terhadap `max_MSE` menyebabkan score compression (mean 3.7–4.5, hampir semua wajah terlihat cantik). Percentile rank menghasilkan distribusi uniform di [1,5], meningkatkan discriminative power. RMSE dipakai untuk konsistensi satuan antar organ. |
| **Aksi** | Update paper §III.D.1: (1) ganti MSE → RMSE, (2) ganti formula linear → percentile rank, (3) tambahkan motivasi: "menghindari efek kompresi akibat outlier tunggal." |

---

## DIFF 3 — Referensi Wajah: Population Average vs Beauty Prototype

| | Detail |
|--|--------|
| **Paper** (§III.D.1) | "Universal Average Face is constructed by computing the coordinate-wise mean of **all** 468 normalized landmarks across the **entire** training set." |
| **Code** ([pseudo_labels.py:147](model/pseudo_labels.py#L147), [pseudo_labels.py:188](model/pseudo_labels.py#L188)) | Menggunakan **Beauty Prototype**: mean dari **top-30% wajah dengan holistic rating tertinggi**, dihitung **per etnicity** (Asian & Caucasian terpisah). |
| **Motivasi** | Population average mencampur wajah cantik + jelek → referensi tidak representatif. Diagnosis menunjukkan Spearman ρ = -0.13 dengan population average (pseudo-label berlawanan arah). Beauty prototype mengisolasi "attractive subspace" sehingga hypothesis averageness tetap valid. |
| **Aksi** | Update paper §III.D.1 dan §III.C (Weakly Supervised Learning): deskripsikan beauty prototype sebagai "Refined Universal Average Face" dari top-k% training faces per ethnicity. Ini juga menjadi kontribusi baru yang perlu dimasukkan ke §I (Introduction) sebagai bagian dari H1 refinement. |

---

## DIFF 4 — H2: Holistic Consistency Filter (Tidak Ada di Paper)

| | Detail |
|--|--------|
| **Paper** (§III.D.2) | Pair (A, B) dipilih berdasarkan **pseudo-score ordering** saja: jika `pseudo(A) > pseudo(B)` untuk suatu organ, model dilatih untuk memprediksi `local_score(A) > local_score(B)`. |
| **Code** ([dataset.py:173](model/dataset.py#L173)) | Tambahan filter: **skip pair jika `holistic_A <= holistic_B`** (H2). Hanya pair yang konsisten antara holistic rating dan pseudo-score ordering yang dimasukkan ke training. |
| **Motivasi** | Tanpa H2, L_rank bisa dilatih untuk memenangkan ranking yang bertentangan dengan L_reg → konflik gradien → λ_rank naik tidak terkontrol. H2 memastikan L_rank dan L_reg tidak mendorong arah berlawanan. |
| **Aksi** | Tambahkan H2 ke paper §III.D.2 sebagai "Consistency-Filtered Pair Sampling": jelaskan bahwa hanya pair dengan holistic ordering konsisten yang digunakan, beserta motivasinya. |

---

## DIFF 5 — GradNorm: 3 Tasks vs 2 Tasks

| | Detail |
|--|--------|
| **Paper** (§III.D.3, §III.D.4) | Formula total loss: $\mathcal{L} = \lambda_1\mathcal{L}_{reg} + \lambda_2\mathcal{L}_{rank} + \lambda_3\mathcal{L}_{div}$ — mengimplikasikan GradNorm mengelola **ketiga** loss secara dinamis. |
| **Code** ([config.py:71](model/config.py#L71), [train.py:239](model/train.py#L239)) | GradNorm hanya mengelola **2 task**: `[L_reg, L_rank]`. `L_div` ditambahkan dengan **bobot tetap** `LDIV_WEIGHT = 0.01` di luar GradNorm: `total_loss += 0.01 * loss_div` |
| **Mengapa berbeda** | `L_div = -Var(scores)` bernilai **negatif**. GradNorm mengasumsikan semua loss positif — memasukkan loss negatif menyebabkan GradNorm tidak stabil (target gradient norm tidak bisa dihitung dengan benar). |
| **Aksi** | Update paper §III.D.3: pisahkan $\mathcal{L}_{div}$ dari GradNorm. Formula baru: $\mathcal{L}_{total} = \text{GradNorm}(\lambda_1\mathcal{L}_{reg}, \lambda_2\mathcal{L}_{rank}) + \lambda_3\mathcal{L}_{div}$ dengan $\lambda_3 = 0.01$ (fixed). Tambahkan kalimat justifikasi: "L_div dikecualikan dari GradNorm karena nilainya negatif, bertentangan dengan asumsi loss positif pada algoritma GradNorm." |

---

## DIFF 6 — L_reg Menggunakan Kedua Wajah dalam Pair

| | Detail |
|--|--------|
| **Paper** (§III.D.3) | $\mathcal{L}_{reg} = \frac{1}{N}\sum(\hat{y}_{global} - y_{gt})^2$ — tidak dispesifikasikan apakah menggunakan satu atau dua wajah per pair. |
| **Code** ([train.py:233](model/train.py#L233)) | `loss_reg = (l_reg(pred_a, rating_a) + l_reg(pred_b, rating_b)) / 2` — L_reg dihitung dari **kedua** wajah A dan B per batch. |
| **Motivasi** | Menggunakan hanya face A membuang 50% ground-truth signal holistic per batch. Menggunakan A+B menggandakan sinyal regresi tanpa menambah data baru. |
| **Aksi** | Update paper §III.D.3: tambahkan keterangan bahwa L_reg dihitung dari kedua wajah dalam setiap pair untuk memaksimalkan utilitas ground-truth label. |

---

## DIFF 7 — Pair Resampling Per Epoch (Tidak Ada di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak disebutkan. |
| **Code** ([train.py:203](model/train.py#L203)) | `pair_ds._pairs = pair_ds._build_pairs()` dipanggil di awal **setiap epoch** — pair di-resample ulang sehingga model tidak overfit pada set pasangan yang tetap. |
| **Motivasi** | Pair tetap menyebabkan model menghafalkan urutan spesifik antar wajah, bukan belajar pola umum. Resampling per epoch setara dengan data augmentation untuk ranking. |
| **Aksi** | Tambahkan ke paper §III.D.2: "Pairs are resampled at the beginning of each epoch to prevent overfitting to a fixed pair ordering and improve generalization." |

---

## DIFF 8 — GAT Multi-Head (Tidak Dispesifikasikan di Paper)

| | Detail |
|--|--------|
| **Paper** (§III.C.2) | Formula GAT standar ditampilkan — tidak menyebutkan jumlah head. Terkesan single-head. |
| **Code** ([config.py:53](model/config.py#L53), [model.py:88](model/model.py#L88)) | `GAT_NUM_HEADS = 4` — 4 parallel attention heads. Output shape `(N, 4, 64)` → flatten → `(N, 256)`. |
| **Aksi** | Update paper §III.C.2: tambahkan "multi-head attention with K=4 heads" dan modifikasi formula aggregasi menjadi multi-head: $h'_i = \|_{k=1}^{K} \sigma\left(\sum_j \alpha_{ij}^k W^k h_j\right)$ |

---

## DIFF 9 — Sub-Graph Connectivity (Tidak Dispesifikasikan di Paper)

| | Detail |
|--|--------|
| **Paper** | Tidak menyebutkan struktur edge dalam organ sub-graph. |
| **Code** ([preprocessing.py](model/preprocessing.py)) | Sub-graph dibangun sebagai **fully connected** (semua node dalam satu organ terhubung ke semua node lain) + **self-loops**. |
| **Aksi** | Tambahkan ke paper §III.B.3: "Each organ sub-graph is constructed as a fully-connected graph with self-loops, allowing every landmark to attend to all others within the same anatomical region." Sertakan justifikasi: memastikan setiap node memiliki akses ke konteks seluruh organ dalam satu GAT layer. |

---

## DIFF 10 — Local Score Validity: Definisi Validasi

| | Detail |
|--|--------|
| **Paper** (§III.E) | "Verifying that facial components geometrically **closer to the Universal Average Face** receive proportionally **higher** aesthetic scores." |
| **Code** ([evaluate.py:116](model/evaluate.py#L116)) | Menghitung **Spearman ρ antara pseudo_scores dan predicted local_scores**. Validasi = semua 5 organ punya ρ > 0. Tidak langsung mengukur jarak ke avg face — melainkan konsistensi dengan pseudo-label. |
| **Implikasi setelah beauty prototype** | Setelah DIFF 3 (beauty prototype), pseudo-scores sekarang: rendah RMSE dari *beauty prototype* → score tinggi. Paper harus diperbarui untuk mencerminkan referensi yang digunakan. |
| **Aksi** | Update paper §III.E: ganti "Universal Average Face" → "Beauty Prototype". Klarifikasi bahwa validasi mengukur Spearman ρ antara predicted organ scores dan pseudo-scores berbasis beauty prototype. |

---

## Ringkasan Aksi Paper

| DIFF | Bagian Paper | Jenis Perubahan |
|------|-------------|-----------------|
| 1 | §III.B, §III.C.1 | Update formula & motivasi 6-dim node feature |
| 2 | §III.D.1 | Ganti MSE→RMSE, linear→percentile, tambah motivasi |
| 3 | §I, §III.C, §III.D.1 | Tambah beauty prototype sebagai kontribusi H1 |
| 4 | §III.D.2 | Tambah H2 consistency filter + motivasi |
| 5 | §III.D.3, §III.D.4 | Pisahkan L_div dari GradNorm, fixed weight λ₃=0.01 |
| 6 | §III.D.3 | Klarifikasi L_reg menggunakan kedua wajah A+B |
| 7 | §III.D.2 | Tambahkan pair resampling per epoch |
| 8 | §III.C.2 | Spesifikasikan K=4 multi-head GAT + formula update |
| 9 | §III.B.3 | Tambahkan fully-connected + self-loop sub-graph |
| 10 | §III.E | Update referensi: avg face → beauty prototype |

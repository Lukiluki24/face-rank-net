# Summary — Perubahan dari Model Initialization ke Versi Saat Ini

Dokumen ini meringkas evolusi FaceRankNet dari commit awal `2c90ba2` ("add: Model Initialization") sampai HEAD (`4b5850c`).

## Riwayat Commit

| Commit | Pesan | Tanggal |
|--------|-------|---------|
| `2c90ba2` | add: Model Initialization | 2026-05-19 |
| `87c930b` | fix: fix bug, add features to gat (katanya membantu) | 2026-05-20 |
| `e09059d` | fix: benerin bug training | 2026-05-20 |
| `4b5850c` | fix: benerin dpd | 2026-05-21 |

## File yang TIDAK Berubah
- [model/model.py](model/model.py) — arsitektur OrganGAT + fusion tetap sama
- [model/evaluate.py](model/evaluate.py)
- [model/organ_indices.py](model/organ_indices.py)

Jadi *arsitektur jaringan* tidak diubah. Semua perubahan ada di sisi **fitur input, pseudo-label, loss-weighting, dan loop training**.

---

## 1. Konfigurasi — [model/config.py](model/config.py)

| Parameter | Awal (`2c90ba2`) | Sekarang | Alasan |
|-----------|------------------|----------|--------|
| `NODE_FEAT_DIM` | `3` (x, y, z) | `6` (x, y, z, Δx, Δy, Δz) | Tambahan deviasi dari Universal Average Face langsung di level input |
| `NUM_TASKS` | `3` (L_reg, L_rank, L_div) | `2` (L_reg, L_rank) | L_div dikeluarkan dari GradNorm karena bernilai negatif |
| `LDIV_WEIGHT` | — | `0.01` | Bobot tetap untuk L_div (di luar GradNorm) |
| `PAIRS_PER_SAMPLE` | `1` | `3` | Lebih banyak pasangan negatif per anchor untuk L_rank |

---

## 2. Preprocessing — [model/preprocessing.py](model/preprocessing.py)

`build_subgraph` dan `build_all_subgraphs` sekarang menerima parameter opsional `avg_face`:

- Jika `avg_face=None` → fitur node tetap 3-dim (koordinat saja, backward compatible).
- Jika `avg_face` diberikan → fitur node menjadi 6-dim: `[koordinat, deviation = coords − avg_face]`.

Tujuan: sinyal **Averageness** sekarang eksplisit di input, tidak perlu lagi diinferensi oleh GAT dari koordinat absolut.

---

## 3. Dataset — [model/dataset.py](model/dataset.py)

### `FaceDataset`
- Tambahan parameter `avg_face: np.ndarray | None`.
- Diteruskan ke `build_all_subgraphs` agar setiap face menghasilkan subgraph dengan fitur 6-dim.

### `PairDataset` (H2 filter)
Saat membentuk pasangan (A, B), ditambahkan guard:

```python
if rating_a <= rating_b:
    continue  # H2: hanya latih L_rank kalau urutan holistic sejalan dengan urutan pseudo
```

Artinya: L_rank hanya dijalankan untuk pasangan yang konsisten antara rating holistic dan pseudo organ — mengurangi konflik antar-loss.

---

## 4. Pseudo Labels — [model/pseudo_labels.py](model/pseudo_labels.py)

### a) MSE → RMSE
`compute_organ_mse` sekarang return `sqrt(mean(diff²))`. Alasan: satuan konsisten antar organ (skala panjang, bukan skala panjang kuadrat).

### b) Per-Ethnicity Average Faces (H1)
Fungsi baru `compute_ethnicity_avg_faces` membuat satu Universal Average Face per kelompok etnis. `compute_all_pseudo_labels` menerima `avg_face_map` + `ethnicity_map` opsional sehingga pseudo-label dihitung relatif terhadap rata-rata kelompok etnis masing-masing wajah (fallback ke global avg untuk unknown ethnicity).

### c) Quality Diagnostic — `validate_pseudo_label_quality`
Fungsi baru yang menghitung **Spearman ρ** antara rata-rata pseudo-score per wajah vs. rating holistik ground-truth. Memberikan warning bila ρ < 0.2 (artinya L_rank kemungkinan konflik dengan L_reg).

---

## 5. Loss — [model/loss.py](model/loss.py)

`GradNorm._get_shared_params` direvisi:

- **Sebelum**: hanya mengambil `input_proj` dari organ pertama. Layer ini cuma menerima gradien dari L_reg → GradNorm tidak balance dengan benar.
- **Sekarang**: mengambil layer terakhir `Linear(32→1)` dari semua `OrganGAT.mlp`. Layer ini merupakan komputasi terakhir sebelum `local_scores`, sehingga **menerima gradien dari L_reg (via global_score) maupun L_rank (via local_scores)** — sesuai prasyarat GradNorm.

---

## 6. Training Loop — [model/train.py](model/train.py)

Perubahan utama:

1. **Pair resampling per epoch**
   ```python
   pair_ds._pairs = pair_ds._build_pairs()
   pair_loader = make_pair_loader(...)
   ```
   Sebelumnya pair_loader dibuat sekali di luar loop. Sekarang dibangun ulang tiap epoch agar model melihat kombinasi (A, B) yang berbeda dan tidak overfit pada satu set ordering tetap.

2. **L_reg pakai dua wajah**
   ```python
   loss_reg = (l_reg(pred_a, rating_a) + l_reg(pred_b, rating_b)) / 2
   ```
   Sebelumnya hanya face A dipakai → 50 % sinyal ground-truth terbuang per batch.

3. **L_div dipisah dari GradNorm**
   ```python
   losses = [loss_reg, loss_rank]                         # GradNorm hanya untuk yang positif
   total_loss = gradnorm.update(losses, optimizer)
   total_loss = total_loss + config.LDIV_WEIGHT * loss_div   # fixed weight
   ```
   L_div bernilai negatif (−Var), tidak cocok dengan asumsi GradNorm bahwa semua loss positif.

4. **Diagnostic call** untuk `validate_pseudo_label_quality` sebelum training mulai.

5. **`avg_face` di-load** dan diteruskan ke train + test dataset (avg_face dihitung dari train-set saja → tidak ada leakage).

---

## 7. Tambahan Dokumen

- [FaceRankNet_Paper.md](FaceRankNet_Paper.md) — draft paper ditambahkan di commit `87c930b`.

---

## Ringkasan Padat

> Arsitektur GAT tidak berubah. Yang berubah: **(1)** node feature jadi 6-dim dengan deviasi dari avg face, **(2)** pseudo-label pakai RMSE + per-etnis avg face + Spearman validation, **(3)** L_rank di-gate dengan urutan rating holistik, **(4)** GradNorm di-anchor di MLP-tail (bukan input_proj) dan hanya menangani L_reg+L_rank, sementara L_div pakai bobot tetap, **(5)** pair di-resample tiap epoch dan L_reg memanfaatkan kedua wajah dalam pair.

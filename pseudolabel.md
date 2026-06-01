
## Ringkasan 9 Metode Pseudo-Label

### 1. `rmse_eth` (Legacy Baseline)

* **Paradigma:** Mengukur kedekatan geometris murni menggunakan nilai akar kuadrat rata-rata eror (*Root Mean Squared Error* / RMSE).
* **Mekanisme:** Menghitung jarak Euclidean organ dari *Beauty Prototype* etnisnya (rata-rata 30% wajah tercantik). Semakin kecil nilai RMSE, semakin tinggi skor akhir wajah tersebut.
* **Karakteristik:** Bersifat seragam (*uniform percentile-ranked*) tetapi rentan terhadap kegagalan jika fitur wajah menarik menyimpang dari satu prototipe kaku.

### 2. `axis_eth` (Current Baseline)

* **Paradigma:** Proyeksi skalar berbasis arah (*Directional Beauty Axis*).
* **Mekanisme:** Membangun vektor arah kecantikan etnis dari rata-rata populasi etnis ($\mu_{\text{eth}}$) menuju *Beauty Prototype* etnis tersebut. Koordinat landmark wajah diproyeksikan secara skalar ke sumbu arah ini.
* **Karakteristik:** Mengakomodasi teori bahwa wajah menarik tidak harus selalu "rata-rata", melainkan wajah yang menyimpang ke arah karikatur kecantikan ideal.

### 3. `axis_eth_gen` (Fine-Grained Axis)

* **Paradigma:** Proyeksi sumbu kecantikan dengan pengondisian demografis silang yang lebih ketat.
* **Mekanisme:** Serupa dengan `axis_eth`, namun populasi rata-rata dan prototipe kecantikannya dihitung secara spesifik per kombinasi etnis dan gender (misalnya: *Asian_Female*, *Caucasian_Male*).
* **Karakteristik:** Berusaha menangkap perbedaan standar kecantikan spesifik gender dalam kelompok ras yang sama.

### 4. `axis_eth_kmeans` (Multi-Prototype Subspaces)

* **Paradigma:** Proyeksi sumbu kecantikan multi-ruang laten.
* **Mekanisme:** Mengelompokkan top-30% wajah tercantik pada setiap etnis menjadi $K=3$ klaster menggunakan algoritma K-Means untuk menghasilkan 3 sub-prototipe (misalnya memisahkan tipe wajah *oval*, *square*, atau *heart*). Setiap wajah diproyeksikan ke ketiga sumbu tersebut dan diambil nilai proyeksi tertingginya.
* **Karakteristik:** Mencegah masalah *feature cancellation* akibat perataan fitur wajah menarik yang memiliki bentuk anatomi bertolak belakang.

### 5. `axis_eth_quantile` (Distribution-Matched Axis)

* **Paradigma:** Rekayasa bentuk distribusi marjinal pasca-proyeksi.
* **Mekanisme:** Mengambil urutan peringkat dari metode `axis_eth` lalu memetakan ulang magnitudonya menggunakan fungsi kuantil agar bentuk distribusinya meniru persis kurva lonceng (*Gaussian-ish*) nilai *ground-truth holistic ratings* manusia.
* **Karakteristik:** Mempertahankan urutan peringkat (*Spearman rho* per organ identik dengan `axis_eth`) tetapi mereduksi konflik optimasi pada model GAT akibat akumulasi skor uniform di batas ekstrem skala 1 dan 5.

### 6. `axis_eth_synthaug` (Data-Expanded Synth Axis)

* **Paradigma:** Augmentasi ruang geometris berbasis MixUp pada level *pseudo-labeling*.
* **Mekanisme:** Mengidentifikasi sampel minoritas ekstrem (bucket Jelek dan Cantik) lalu melakukan pencampuran linear koordinat landmark wajah intern dalam bucket tersebut untuk mensintesis 1000 wajah baru (meningkatkan total data latihan dari 4400 menjadi 5400). Sumbu kecantikan dihitung ulang pada dataset augmented ini, diikuti proyeksi dan *quantile remap*.
* **Karakteristik:** Memberikan variasi geometri baru bagi kelas minoritas guna mencegah GAT mengalami *overfitting* saat fase training nanti.

### 7. `symmetry` (Bilateral Structural Prior)

* **Paradigma:** Sinyal struktural internal non-prototipe berbasis simetri wajah.
* **Mekanisme:** Mencerminkan koordinat landmark secara horizontal melewati garis tengah wajah ($x \rightarrow -x$). Mengukur jarak rata-rata tetangga terdekat (*nearest-neighbor distance*) antara organ asli dengan organ hasil refleksi sumbu-X pasangannya.
* **Karakteristik:** Eror refleksi yang kecil diubah menjadi skor kecantikan yang tinggi melalui konversi peringkat persentil balik.

### 8. `canons` (Neoclassical Proportion Prior)

* **Paradigma:** Sinyal struktural ideal abad pertengahan.
* **Mekanisme:** Mengevaluasi deviasi relatif wajah terhadap 6 aturan proporsi klasik (*Neoclassical Canons*), meliputi *Rule of Thirds* (pembagian vertikal wajah), konsistensi lebar mata kiri-kanan, rasio jarak antar mata, rasio lebar mulut-mata, rasio hidung-mata, hingga proporsi emas wajah (*Golden Ratio* $\approx 1.618$).
* **Karakteristik:** Eror akumulasi dari aturan-aturan yang menyangkut organ tertentu digabungkan dan diubah menjadi skor 1-5. Sinyal ini murni berbasis aturan geometris kaku, bukan statistik populasi.

### 9. `axis_quantile_canons_sym` (Hybrid Blender)

* **Paradigma:** Kombinasi ensemble multi-teori (Statistik + Struktural).
* **Mekanisme:** Menggabungkan tiga pilar penilaian melalui rata-rata tertimbang (*weighted average*) langsung pada level skor organ. Bobot yang dialokasikan adalah 50% untuk metode geometri terbaik (`axis_eth_quantile`), 25% untuk `canons`, dan 25% untuk `symmetry`.
* **Karakteristik:** Berusaha menguji apakah aturan struktural kaku mampu mengoreksi atau melengkapi sinyal kecantikan yang dipelajari dari ruang laten etnis populasi.

---

## Tabel Perbandingan Karakteristik Metode

| No | Nama Metode | Basis Teori Matematika | Granularitas Kelompok | Bentuk Distribusi Akhir |
| --- | --- | --- | --- | --- |
| 1 | `rmse_eth` | Jarak Euclidean dari Titik Ideal | Per Etnis | Uniform $[1, 5]$ |
| 2 | `axis_eth` | Proyeksi Vektor Arah Estetika | Per Etnis | Uniform $[1, 5]$ |
| 3 | `axis_eth_gen` | Proyeksi Vektor Arah Estetika | Etnis $\times$ Gender | Uniform $[1, 5]$ |
| 4 | `axis_eth_kmeans` | Proyeksi Maksimum Multi-Sumbu | Per Etnis (3 Klaster) | Uniform $[1, 5]$ |
| 5 | `axis_eth_quantile` | Proyeksi Sumbu + *Empirical* CDF Match | Per Etnis | Meniru *Ground Truth* (Kurva Lonceng) |
| 6 | `axis_eth_synthaug` | Proyeksi Sumbu + MixUp Geometris | Per Etnis (Dataset 5400 baris) | Meniru *Ground Truth* (Kurva Lonceng) |
| 7 | `symmetry` | Eror Nearest-Neighbor Refleksi X | Global (Tanpa Kelompok) | Uniform $[1, 5]$ |
| 8 | `canons` | Eror Akumulasi 6 Aturan Proporsi Klasik | Global (Tanpa Kelompok) | Uniform $[1, 5]$ |
| 9 | `axis_quantile_canons_sym` | Rata-rata Tertimbang Linier Ensemble | Campuran | Terdistorsi (Pencampuran Bentuk Skala) |
# PRODUCT REQUIREMENTS DOCUMENT (PRD)

## X-Ray Steganography Research Platform

**Versi:** 1.0
**Status:** Draft
**Jenis proyek:** Research Prototype
**Platform:** Desktop / Local Application
**Bahasa implementasi:** Python
**Domain:** Image Processing, Steganography, Medical Imaging

---

# 1. Ringkasan Proyek

## 1.1. Latar Belakang

Steganografi merupakan teknik menyisipkan informasi rahasia ke dalam media digital tanpa menampilkan keberadaan pesan secara langsung. Pada citra medis, steganografi dapat digunakan untuk mengeksplorasi penyisipan metadata atau informasi terkait pemeriksaan ke dalam citra.

Penelitian ini berfokus pada evaluasi komparatif tiga metode steganografi citra:

1. Least Significant Bit (LSB).
2. Content-Adaptive LSB.
3. Discrete Cosine Transform (DCT).

Ketiga metode akan diuji menggunakan citra X-ray dada dan pesan pasien sintetis. Evaluasi dilakukan untuk mengukur kualitas visual citra setelah penyisipan serta ketahanan ekstraksi pesan setelah citra mengalami kompresi JPEG.

Proyek ini menghasilkan aplikasi eksperimen yang mampu menjalankan proses embedding, extraction, simulasi kompresi JPEG, perhitungan metrik evaluasi, dan penyimpanan hasil eksperimen.

## 1.2. Tujuan Produk

Membangun platform eksperimen yang memungkinkan peneliti membandingkan tiga metode steganografi dalam kondisi pengujian yang terkontrol dan dapat direproduksi.

Tujuan utamanya adalah:

* Mengimplementasikan tiga metode steganografi.
* Menyisipkan pesan sintetis ke dalam citra X-ray dada.
* Mengukur distorsi citra akibat penyisipan.
* Menguji keberhasilan ekstraksi setelah kompresi JPEG.
* Membandingkan pengaruh ukuran payload dan kualitas JPEG.
* Menghasilkan tabel dan visualisasi hasil eksperimen.
* Mendukung analisis ilmiah melalui pencatatan konfigurasi dan parameter eksperimen.

## 1.3. Judul Penelitian

**Analisis Komparatif Metode LSB, Content-Adaptive LSB, dan DCT untuk Penyisipan Data Pasien pada Citra X-Ray Dada: Evaluasi Kualitas Citra dan Ketahanan Ekstraksi terhadap Kompresi JPEG**

Judul ini merupakan judul kerja dan dapat disesuaikan setelah studi literatur serta penetapan research gap.

---

# 2. Target Pengguna

## 2.1. Pengguna Utama

Mahasiswa atau peneliti yang melakukan eksperimen steganografi citra digital.

## 2.2. Kebutuhan Pengguna

Pengguna membutuhkan sistem yang dapat:

* Menguji algoritma tanpa harus menjalankan setiap eksperimen secara manual.
* Menggunakan dataset citra yang sama untuk seluruh metode.
* Mengontrol ukuran pesan yang disisipkan.
* Mengontrol kualitas kompresi JPEG.
* Memeriksa keberhasilan pemulihan pesan.
* Membandingkan hasil metode secara kuantitatif.
* Mengekspor hasil agar dapat digunakan dalam laporan penelitian.

---

# 3. Ruang Lingkup Produk

## 3.1. Fitur yang Termasuk dalam MVP

MVP adalah versi awal aplikasi yang harus berfungsi sebelum pengembangan lanjutan dilakukan.

| ID    | Fitur                                                     | Prioritas   |
| ----- | --------------------------------------------------------- | ----------- |
| FR-01 | Memuat citra X-ray                                        | Must have   |
| FR-02 | Memasukkan pesan sintetis                                 | Must have   |
| FR-03 | Embedding menggunakan LSB                                 | Must have   |
| FR-04 | Extraction menggunakan LSB                                | Must have   |
| FR-05 | Embedding dan extraction menggunakan Content-Adaptive LSB | Must have   |
| FR-06 | Embedding dan extraction menggunakan DCT                  | Must have   |
| FR-07 | Simulasi kompresi JPEG                                    | Must have   |
| FR-08 | Penghitungan PSNR, SSIM, dan MSE                          | Must have   |
| FR-09 | Penghitungan BER dan keberhasilan ekstraksi               | Must have   |
| FR-10 | Pencatatan waktu embedding dan extraction                 | Must have   |
| FR-11 | Penyimpanan hasil eksperimen dalam CSV                    | Must have   |
| FR-12 | Perbandingan hasil antarmetode                            | Must have   |
| FR-13 | Visualisasi grafik hasil eksperimen                       | Should have |
| FR-14 | Pengujian berbagai ukuran payload                         | Must have   |
| FR-15 | Pengujian berbagai tingkat kualitas JPEG                  | Must have   |

## 3.2. Fitur di Luar Ruang Lingkup MVP

Fitur berikut tidak menjadi bagian dari versi pertama:

* Penggunaan data identitas pasien asli.
* Integrasi dengan rumah sakit atau sistem rekam medis elektronik.
* Integrasi langsung dengan PACS.
* Penggunaan untuk diagnosis klinis.
* Sistem autentikasi pengguna dan manajemen akun.
* Pengembangan metode berbasis GAN atau reinforcement learning.
* Steganalisis berbasis deep learning.
* Pengujian seluruh jenis serangan citra.
* Penggunaan DICOM sebagai format eksperimen utama sebelum pipeline pemrosesannya tervalidasi.

Fitur di luar ruang lingkup dapat dipertimbangkan setelah eksperimen dasar berhasil.

---

# 4. Deskripsi Metode

## 4.1. LSB Konvensional

LSB menyisipkan bit pesan pada bit paling tidak signifikan dari piksel citra.

Karakteristik implementasi:

* Menggunakan aturan penentuan lokasi yang sederhana.
* Memiliki proses embedding dan extraction.
* Menyimpan informasi panjang pesan atau metadata yang diperlukan untuk ekstraksi.
* Menyediakan pemeriksaan kapasitas sebelum embedding.
* Menggunakan representasi citra yang konsisten.

Untuk baseline awal, implementasi dapat menggunakan citra grayscale 8-bit.

## 4.2. Content-Adaptive LSB

Content-Adaptive LSB menentukan lokasi penyisipan berdasarkan karakteristik citra.

Implementasi penelitian diarahkan untuk mereplikasi metode dari paper utama yang menggunakan saliency fusion dan Ant Colony Optimization (ACO), sejauh detail algoritma dan implementasinya tersedia.

Komponen yang perlu ditentukan meliputi:

* Ekstraksi fitur atau saliency.
* Pembentukan peta prioritas lokasi.
* Mekanisme pemilihan lokasi embedding.
* Penyimpanan atau rekonstruksi informasi yang diperlukan untuk extraction.
* Penanganan kapasitas dan batas distorsi.

Jika implementasi menggunakan algoritma adaptif yang disederhanakan, sistem harus mencatat bahwa metode tersebut bukan replikasi persis dari paper utama.

## 4.3. DCT-Based Steganography

DCT mengubah blok citra dari domain spasial ke domain frekuensi. Pesan disisipkan menggunakan aturan modifikasi koefisien tertentu.

Implementasi wajib menetapkan:

* Ukuran blok.
* Pemilihan koefisien.
* Aturan penyisipan bit.
* Prosedur ekstraksi.
* Kapasitas yang tersedia.
* Penanganan perubahan koefisien setelah kompresi JPEG.

DCT bukan satu algoritma steganografi tunggal. Oleh karena itu, implementasi harus menyebutkan skema embedding dan extraction yang digunakan.

---

# 5. Alur Kerja Utama

## 5.1. Alur Eksperimen

1. Pengguna memilih dataset citra X-ray.
2. Sistem memvalidasi format dan karakteristik citra.
3. Pengguna menentukan pesan sintetis.
4. Pengguna memilih metode steganografi.
5. Pengguna menentukan ukuran payload.
6. Sistem menjalankan embedding.
7. Sistem menyimpan stego-image.
8. Sistem menjalankan extraction tanpa serangan.
9. Sistem memvalidasi pesan hasil ekstraksi.
10. Sistem menghitung metrik kualitas citra.
11. Sistem menjalankan kompresi JPEG jika pengujian serangan diaktifkan.
12. Sistem menjalankan extraction terhadap citra hasil kompresi.
13. Sistem menghitung BER dan extraction success rate.
14. Sistem mencatat waktu komputasi.
15. Sistem menyimpan hasil eksperimen.
16. Sistem menghasilkan tabel dan grafik perbandingan.

## 5.2. Diagram Alur

```text
             Input X-Ray Image
                     |
                     v
             Image Validation
                     |
                     v
             Synthetic Message
                     |
                     v
              Method Selection
                     |
          +----------+----------+
          |          |          |
          v          v          v
         LSB      Adaptive      DCT
                    LSB
          |          |          |
          +----------+----------+
                     |
                     v
                Stego Image
                     |
                     v
          Baseline Extraction Test
                     |
                     v
              JPEG Compression
                     |
                     v
              Extraction Test
                     |
                     v
             Metrics Calculation
                     |
                     v
              Results and Graphs
```

---

# 6. Persyaratan Fungsional

## FR-01 — Memuat Citra

Sistem harus menerima citra dari direktori lokal.

Persyaratan:

* Mendukung format yang ditentukan dalam konfigurasi eksperimen.
* Memvalidasi ukuran citra.
* Memeriksa apakah citra dapat dibaca.
* Menyimpan identitas file untuk pelacakan hasil.
* Mempertahankan citra sumber tanpa modifikasi.

Format utama MVP adalah PNG grayscale 8-bit. Dukungan format lain dapat ditambahkan setelah pipeline validasi tersedia.

**Kriteria penerimaan:**

* Citra valid dapat dimuat.
* Citra rusak menghasilkan pesan kesalahan yang jelas.
* Citra referensi tidak tertimpa selama eksperimen.

## FR-02 — Memasukkan Pesan

Sistem harus menerima pesan sintetis yang akan disisipkan.

Persyaratan:

* Mendukung teks UTF-8.
* Menghitung ukuran pesan dalam byte.
* Mengubah pesan menjadi representasi bit.
* Menyimpan pesan referensi untuk pemeriksaan.
* Menolak payload yang melebihi kapasitas metode.

**Kriteria penerimaan:**

* Panjang pesan tercatat dengan benar.
* Pesan dapat direkonstruksi setelah extraction tanpa serangan.
* Payload yang melebihi kapasitas ditolak sebelum embedding.

## FR-03 — Embedding

Sistem harus menyediakan fungsi embedding untuk ketiga metode.

Setiap fungsi menerima:

* Citra referensi.
* Pesan.
* Konfigurasi metode.

Setiap fungsi menghasilkan:

* Stego-image.
* Informasi konfigurasi.
* Waktu embedding.
* Status keberhasilan.

**Kriteria penerimaan:**

* Ketiga metode dapat dijalankan secara independen.
* Setiap metode memiliki prosedur extraction yang sesuai.
* Hasil embedding dapat disimpan tanpa mengubah citra sumber.

## FR-04 — Extraction

Sistem harus mengekstrak pesan dari stego-image.

**Kriteria penerimaan:**

* Pesan dapat diekstrak tanpa serangan jika implementasi valid.
* Pesan hasil extraction dibandingkan dengan pesan asli.
* Kegagalan extraction dicatat sebagai hasil eksperimen, bukan disembunyikan sebagai error yang tidak tercatat.
* Decoder menggunakan informasi yang tersedia secara realistis bagi penerima.

## FR-05 — Kompresi JPEG

Sistem harus dapat melakukan simulasi kompresi JPEG terhadap stego-image.

Parameter awal:

* Quality 95.
* Quality 90.
* Quality 85.
* Quality 80.
* Quality 75.

Nilai tersebut merupakan parameter eksperimen yang dapat dikonfigurasi.

**Kriteria penerimaan:**

* Setiap kualitas menghasilkan file uji yang dapat dibaca.
* File hasil kompresi disimpan terpisah.
* Parameter kompresi dicatat.
* Citra hasil kompresi tidak menimpa citra sumber atau stego-image awal.

Catatan: implementasi JPEG harus mencatat library dan parameter yang digunakan karena hasil kompresi dapat berbeda antarimplementasi.

## FR-06 — Perhitungan Metrik

Sistem harus menghitung metrik evaluasi berikut.

| Metrik                  | Tujuan                                                       |
| ----------------------- | ------------------------------------------------------------ |
| MSE                     | Mengukur rata-rata kesalahan piksel                          |
| PSNR                    | Mengukur kualitas rekonstruksi berdasarkan error piksel      |
| SSIM                    | Mengukur kemiripan struktural citra                          |
| BER                     | Mengukur proporsi bit pesan yang salah                       |
| Extraction Success Rate | Mengukur proporsi pesan yang berhasil dipulihkan secara utuh |
| Embedding Time          | Mengukur waktu penyisipan                                    |
| Extraction Time         | Mengukur waktu ekstraksi                                     |
| Payload                 | Mengukur ukuran pesan yang disisipkan                        |

Sistem harus membedakan tiga perbandingan kualitas citra:

1. Cover versus stego sebelum kompresi.
2. Stego sebelum versus sesudah kompresi.
3. Cover versus citra setelah embedding dan kompresi.

Setiap nilai PSNR, MSE, atau SSIM harus menyimpan informasi mengenai pasangan citra yang dibandingkan.

## FR-07 — Penyimpanan Hasil

Sistem harus menyimpan hasil eksperimen dalam format CSV.

Kolom minimum:

```text
experiment_id
image_id
method
payload_bytes
payload_bpp
jpeg_quality
mse_cover_stego
psnr_cover_stego
ssim_cover_stego
mse_stego_compressed
psnr_stego_compressed
ssim_stego_compressed
ber
extraction_success
embedding_time_seconds
extraction_time_seconds
```

Kolom tambahan dapat mencatat versi algoritma, konfigurasi, dan alasan kegagalan.

---

# 7. Persyaratan Nonfungsional

## NFR-01 — Reproducibility

Eksperimen harus dapat diulang dengan konfigurasi yang sama.

Sistem perlu mencatat:

* Versi kode.
* Versi library.
* Dataset dan identitas citra.
* Parameter metode.
* Ukuran payload.
* Parameter JPEG.
* Timestamp eksperimen.

Jika suatu algoritma menggunakan randomisasi, seed harus dicatat agar eksperimen dapat direproduksi sejauh implementasinya memungkinkan.

## NFR-02 — Integritas Data

Citra sumber tidak boleh dimodifikasi secara langsung.

Seluruh file keluaran harus disimpan pada direktori terpisah.

Pesan asli dan hasil ekstraksi harus dapat dibandingkan menggunakan pemeriksaan byte dan hash.

## NFR-03 — Efisiensi

Sistem harus dapat menjalankan eksperimen pada komputer pribadi tanpa memerlukan GPU khusus untuk metode baseline.

Waktu komputasi harus diukur secara konsisten menggunakan prosedur yang terdokumentasi.

## NFR-04 — Modularitas

Implementasi ketiga metode harus dipisahkan ke dalam modul yang berbeda agar dapat diuji, diganti, atau dikembangkan tanpa mengubah keseluruhan sistem.

## NFR-05 — Privasi

MVP hanya menggunakan data sintetis.

Jika dataset berisi informasi identitas pasien, akses dan penggunaannya harus mengikuti ketentuan dataset serta persyaratan privasi yang berlaku.

## NFR-06 — Batas Penggunaan

Sistem merupakan prototipe penelitian.

Sistem tidak boleh diposisikan sebagai perangkat klinis, sistem rekam medis produksi, atau alat yang menjamin keamanan informasi pasien.

---

# 8. Metrik dan Definisi Evaluasi

## 8.1. Mean Squared Error

Untuk citra referensi \(I\) dan citra hasil \(K\):

$$
MSE = \frac{1}{MN}
\sum_{i=1}^{M}\sum_{j=1}^{N}
(I_{ij}-K_{ij})^2
$$

MSE yang lebih rendah menunjukkan error piksel yang lebih kecil.

## 8.2. Peak Signal-to-Noise Ratio

Untuk citra grayscale 8-bit:

$$
PSNR = 10\log_{10}\left(\frac{255^2}{MSE}\right)
$$

PSNR yang lebih tinggi menunjukkan error piksel yang lebih kecil.

Jika MSE bernilai nol, PSNR dinyatakan tak terhingga.

## 8.3. Structural Similarity Index

SSIM digunakan untuk mengevaluasi kemiripan struktural antara dua citra.

Nilainya harus dihitung menggunakan implementasi yang konsisten, dengan parameter yang dicatat.

## 8.4. Bit Error Rate

$$
BER =
\frac{\text{Jumlah bit salah}}
{\text{Jumlah bit yang dibandingkan}}
$$

BER hanya dapat dihitung secara bermakna jika pesan hasil ekstraksi dapat disejajarkan dengan pesan asli.

Jika ekstraksi gagal total atau panjang pesan tidak diketahui, sistem harus mencatat kegagalan tersebut secara eksplisit. Sistem tidak boleh secara otomatis memberikan BER nol atau mengabaikan kasus gagal.

## 8.5. Extraction Success Rate

$$
ESR =
\frac{\text{Jumlah pesan yang pulih secara utuh}}
{\text{Jumlah percobaan ekstraksi}}
\times 100\%
$$

Pesan dianggap berhasil dipulihkan jika byte hasil ekstraksi identik dengan byte pesan asli.

## 8.6. Payload

Untuk citra berukuran \(W \times H\):

$$
BPP = \frac{\text{Jumlah bit payload}}{W \times H}
$$

Header dan metadata harus dicatat secara terpisah. Jika kapasitas efektif yang tersedia bagi pesan pengguna dilaporkan, overhead protokol tidak boleh dihitung sebagai payload pengguna.

---

# 9. Rancangan Eksperimen

## 9.1. Variabel Independen

1. Metode steganografi.
2. Ukuran payload.
3. Kualitas kompresi JPEG.

## 9.2. Variabel Dependen

1. PSNR.
2. SSIM.
3. MSE.
4. BER.
5. Extraction Success Rate.
6. Waktu embedding.
7. Waktu extraction.

## 9.3. Variabel yang Dikontrol

* Dataset citra.
* Citra sumber.
* Pesan rahasia.
* Ukuran payload untuk perbandingan utama.
* Prosedur kompresi.
* Prosedur pengukuran.
* Kondisi baseline.

## 9.4. Skenario Pengujian

### Skenario A — Baseline

Setiap metode diuji tanpa kompresi JPEG.

Tujuan:

* Memvalidasi embedding.
* Memvalidasi extraction.
* Mengukur distorsi awal.
* Mengukur waktu komputasi awal.

### Skenario B — Ketahanan terhadap JPEG

Stego-image dikompresi dengan beberapa tingkat kualitas JPEG.

Tujuan:

* Mengukur BER.
* Mengukur keberhasilan ekstraksi.
* Mengukur perubahan kualitas citra.

### Skenario C — Pengaruh Payload

Ketiga metode diuji pada beberapa ukuran payload.

Tujuan:

* Mengukur perubahan kualitas citra.
* Mengukur pengaruh kapasitas terhadap ketahanan ekstraksi.
* Membandingkan trade-off antarmetode.

### Skenario D — Evaluasi Lintas Citra

Pengujian diulang pada sejumlah citra X-ray yang sama untuk ketiga metode.

Tujuan:

* Menilai konsistensi hasil.
* Mengurangi ketergantungan kesimpulan pada satu citra.
* Mendukung analisis statistik.

Jumlah citra final harus ditentukan berdasarkan cakupan penelitian dan kebutuhan analisis. Jumlah untuk prototipe awal tidak boleh otomatis dianggap cukup untuk penelitian final.

---

# 10. Kriteria Keberhasilan MVP

MVP dianggap selesai apabila seluruh kriteria berikut terpenuhi.

| ID    | Kriteria penerimaan                                                             |
| ----- | ------------------------------------------------------------------------------- |
| AC-01 | Sistem dapat membaca citra uji yang valid                                       |
| AC-02 | LSB berhasil menyisipkan dan mengekstrak pesan tanpa serangan                   |
| AC-03 | Content-Adaptive LSB berhasil menyisipkan dan mengekstrak pesan tanpa serangan  |
| AC-04 | DCT berhasil menyisipkan dan mengekstrak pesan tanpa serangan                   |
| AC-05 | Sistem dapat menjalankan kompresi JPEG pada seluruh kualitas yang dikonfigurasi |
| AC-06 | Sistem dapat menghitung PSNR, SSIM, MSE, dan BER sesuai definisi                |
| AC-07 | Kegagalan extraction tercatat secara eksplisit                                  |
| AC-08 | Hasil eksperimen tersimpan dalam CSV                                            |
| AC-09 | Setiap hasil memiliki identitas citra, metode, payload, dan parameter JPEG      |
| AC-10 | Eksperimen dapat diulang menggunakan konfigurasi yang sama                      |
| AC-11 | Sistem menghasilkan ringkasan perbandingan ketiga metode                        |
| AC-12 | Citra sumber tidak tertimpa selama eksperimen                                   |

Keberhasilan MVP tidak berarti satu metode harus unggul. MVP berhasil jika pengujian dapat dijalankan secara benar dan hasilnya dapat dianalisis.

---

# 11. Struktur Modul Perangkat Lunak

```text
steganografi-xray/
│
├── data/
│   ├── cover/
│   └── secret/
│
├── outputs/
│   ├── stego/
│   ├── compressed/
│   └── recovered/
│
├── src/
│   ├── lsb.py
│   ├── adaptive_lsb.py
│   ├── dct.py
│   ├── jpeg_attack.py
│   ├── metrics.py
│   ├── dataset.py
│   ├── experiment.py
│   └── reporting.py
│
├── tests/
│   ├── test_lsb.py
│   ├── test_adaptive_lsb.py
│   ├── test_dct.py
│   └── test_metrics.py
│
├── results/
│   ├── experiment_results.csv
│   └── figures/
│
├── requirements.txt
├── README.md
└── config.yaml
```

### Tanggung jawab modul

* `lsb.py`: embedding dan extraction LSB.
* `adaptive_lsb.py`: embedding dan extraction Content-Adaptive LSB.
* `dct.py`: embedding dan extraction DCT.
* `jpeg_attack.py`: simulasi kompresi JPEG.
* `metrics.py`: perhitungan metrik.
* `dataset.py`: pemuatan dan validasi citra.
* `experiment.py`: orkestrasi eksperimen.
* `reporting.py`: tabel dan visualisasi.
* `config.yaml`: parameter eksperimen.

---

# 12. Tahapan Pengembangan

## Milestone 1 — Baseline LSB

Target:

* Struktur proyek tersedia.
* Citra dapat dimuat.
* LSB dapat menyisipkan dan mengekstrak pesan.
* Metrik kualitas dasar berfungsi.

## Milestone 2 — Content-Adaptive LSB

Target:

* Metode adaptif terimplementasi.
* Prosedur ekstraksi tervalidasi.
* Konfigurasi algoritma terdokumentasi.

## Milestone 3 — DCT

Target:

* Skema embedding DCT terimplementasi.
* Prosedur ekstraksi berfungsi.
* Kapasitas dan parameter metode terdokumentasi.

## Milestone 4 — Eksperimen JPEG

Target:

* Kompresi JPEG dapat dijalankan secara otomatis.
* BER dan extraction success rate dapat dihitung.
* Hasil tersimpan dalam CSV.

## Milestone 5 — Analisis

Target:

* Grafik perbandingan tersedia.
* Hasil setiap metode dapat diringkas.
* Analisis trade-off dapat disusun.
* Laporan eksperimen dapat direproduksi.

---

# 13. Risiko dan Mitigasi

| Risiko                                             | Dampak                       | Mitigasi                                           |
| -------------------------------------------------- | ---------------------------- | -------------------------------------------------- |
| Detail algoritma paper tidak lengkap               | Replikasi tidak identik      | Dokumentasikan asumsi dan perbedaan implementasi   |
| Metode DCT tidak dapat menampung payload yang sama | Perbandingan tidak adil      | Tentukan rentang payload bersama                   |
| Extraction gagal sebelum JPEG                      | Hasil robustness tidak valid | Wajibkan baseline extraction berhasil              |
| Kompresi mengubah representasi citra               | Metrik tidak sebanding       | Tetapkan pipeline format dan preprocessing         |
| Hasil bergantung pada satu citra                   | Generalisasi lemah           | Uji pada beberapa citra                            |
| BER tidak dapat dihitung setelah kegagalan total   | Evaluasi bias                | Catat extraction failure secara eksplisit          |
| Data pasien sensitif                               | Risiko privasi               | Gunakan data sintetis dan dataset sesuai ketentuan |
| Hasil visual baik tetapi pesan rusak               | Kesimpulan keliru            | Pisahkan metrik kualitas citra dan ketahanan pesan |

---

# 14. Deliverables

Produk akhir penelitian terdiri atas:

1. Source code tiga metode steganografi.
2. Dataset eksperimen yang penggunaannya sesuai ketentuan.
3. Konfigurasi eksperimen.
4. Hasil embedding dan extraction.
5. Hasil pengujian kompresi JPEG.
6. CSV hasil eksperimen.
7. Grafik dan tabel perbandingan.
8. Dokumentasi instalasi dan reproduksi eksperimen.
9. Laporan metodologi dan hasil penelitian.

---

# 15. Batasan Interpretasi Hasil

Penelitian ini mengevaluasi kualitas citra digital dan ketahanan ekstraksi pesan pada kondisi eksperimen yang ditetapkan.

Hasil PSNR, SSIM, atau MSE tidak dengan sendirinya membuktikan bahwa citra tetap aman digunakan untuk diagnosis.

Demikian pula, keberhasilan ekstraksi setelah JPEG tidak dengan sendirinya membuktikan keamanan steganografi terhadap steganalisis.

Kesimpulan harus dibatasi pada metode, dataset, payload, format citra, dan skenario serangan yang benar-benar diuji.

---

# 16. Definition of Done

Proyek dinyatakan selesai apabila:

* Ketiga metode telah diimplementasikan dan diuji.
* Seluruh metode memiliki prosedur extraction yang sesuai.
* Baseline tanpa serangan berhasil dijalankan.
* Pengujian JPEG dapat direproduksi.
* Metrik dihitung dengan definisi yang konsisten.
* Hasil disimpan secara terstruktur.
* Perbandingan dilakukan pada kondisi eksperimen yang sebanding.
* Seluruh keterbatasan implementasi didokumentasikan.
* Kesimpulan penelitian didukung oleh hasil pengujian aktual.

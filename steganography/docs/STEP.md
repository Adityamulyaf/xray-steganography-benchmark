# PANDUAN TAHAPAN PENGERJAAN PROYEK (STEP-BY-STEP IMPLEMENTATION PLAN)

## X-Ray Steganography Research Platform
**Referensi Dokumen:** PRD.md  
**Versi:** 1.0  
**Domain:** Pengolahan Citra Digital, Steganografi, Citra Medis X-Ray  

---

## 1. Ikhtisar Alur Kerja

Dokumen ini menjabarkan peta jalan teknis dan tahapan eksekusi langkah demi langkah untuk membangun platform eksperimen steganografi komparatif antara metode:
1. **LSB Konvensional** (*Spatial Domain Baseline*)
2. **Content-Adaptive LSB** (*Adaptive Spatial Domain*)
3. **Discrete Cosine Transform (DCT)** (*Frequency Domain*)

Implementasi mengadopsi pendekatan modular berorientasi pengujian (*test-driven & milestone-based*), memastikan setiap metode divalidasi pada skenario baseline sebelum diuji terhadap serangan kompresi JPEG.

---

## 2. Rincian Tahapan Pengerjaan

### **Fase 0: Inisialisasi Lingkungan & Struktur Proyek (Scaffolding)**
> **Fokus:** Menyiapkan virtual environment, dependensi library, struktur direktori terisolasi, dan konfigurasi eksperimen.

1. **Setup File Dependensi (`requirements.txt`):**
   - Menyiapkan pustaka inti:
     - `numpy` & `scipy`: Operasi array numerik dan transformasi frekuensi (2D DCT/IDCT).
     - `opencv-python` & `pillow`: Operasi citra I/O dan simulasi kompresi JPEG.
     - `scikit-image`: Perhitungan metrik kesamaan struktural (SSIM).
     - `pandas`: Manajemen dan ekspor data tabular hasil eksperimen ke CSV.
     - `matplotlib` & `seaborn`: Pembuatan visualisasi grafik analitis.
     - `pyyaml`: Pembaca file konfigurasi parameter terpusat.
     - `pytest`: Kerangka kerja pengujian unit (unit testing).

2. **Membangun Struktur Direktori Proyek:**
   Sesuai arsitektur Bab 11 PRD:
   ```text
   steganography/
   ├── config.yaml              # Konfigurasi parameter eksperimen
   ├── requirements.txt         # Daftar dependensi library Python
   ├── README.md                # Dokumentasi petunjuk penggunaan
   ├── data/
   │   ├── cover/               # Citra referensi X-ray dada (PNG grayscale 8-bit)
   │   └── secret/              # Pesan rahasia sintetis (.txt / .json)
   ├── outputs/
   │   ├── stego/               # Citra hasil embedding
   │   ├── compressed/          # Citra stego pasca-kompresi JPEG
   │   └── recovered/           # Teks hasil ekstraksi
   ├── src/
   │   ├── __init__.py
   │   ├── dataset.py           # I/O citra & generator pesan
   │   ├── metrics.py           # Implementasi MSE, PSNR, SSIM, BER, ESR
   │   ├── lsb.py               # Algoritma LSB konvensional
   │   ├── adaptive_lsb.py      # Algoritma Content-Adaptive LSB
   │   ├── dct.py               # Algoritma DCT domain frekuensi
   │   ├── jpeg_attack.py       # Simulasi kompresi JPEG berbagai kualitas
   │   ├── experiment.py        # Runner dan orkestrator eksperimen
   │   └── reporting.py         # Visualisasi grafik dan tabel komparasi
   ├── tests/
   │   ├── test_metrics.py
   │   ├── test_lsb.py
   │   ├── test_adaptive_lsb.py
   │   └── test_dct.py
   └── results/
       ├── experiment_results.csv
       └── figures/
   ```

3. **Membuat Konfigurasi Eksperimen Terpusat (`config.yaml`):**
   - Daftar ukuran payload: misal `[64, 128, 256, 512, 1024]` byte.
   - Daftar kualitas JPEG: `[95, 90, 85, 80, 75]`.
   - Seed acak untuk memastikan keterulangan (*reproducibility*).
   - Jalur direktori input dan output.

---

### **Fase 1: Fondasi Pipeline Data & Metrik Evaluasi**
> **Fokus:** Memastikan modul input/output citra aman dan metrik evaluasi matematis teruji sebelum algoritma steganografi dibangun (FR-01, FR-02, FR-06).

1. **Modul Dataset & Validasi Citra (`src/dataset.py`):**
   - Validasi citra input: format PNG, mode grayscale 8-bit, resolusi citra, dan pencegahan modifikasi (*read-only reference*).
   - Generator & pemuat pesan pasien sintetis (UTF-8) dengan pencatatan panjang byte.
   - Fungsi konversi pesan teks ke representasi bit stream dan sebaliknya.
   - Pengecekan batas kapasitas teoritis citra sebelum embedding.

2. **Modul Metrik Kualitas & Ekstraksi (`src/metrics.py`):**
   - **MSE (Mean Squared Error):**
     $$MSE = \frac{1}{MN} \sum_{i=1}^{M}\sum_{j=1}^{N} (I_{ij}-K_{ij})^2$$
   - **PSNR (Peak Signal-to-Noise Ratio):**
     $$PSNR = 10\log_{10}\left(\frac{255^2}{MSE}\right) \quad (\text{jika } MSE = 0 \to \infty)$$
   - **SSIM (Structural Similarity Index):** Menggunakan `skimage.metrics.structural_similarity`.
   - **BER (Bit Error Rate):** Proporsi bit pesan yang salah dibandingkan panjang bit yang diuji.
   - **Extraction Success Rate (ESR):** Status boolean keberhasilan pemulihan pesan secara identik 100%.
   - **Payload Density (BPP):** Rasio bit payload terhadap total piksel citra ($W \times H$).

3. **Pengujian Unit Metrik (`tests/test_metrics.py`):**
   - Verifikasi bahwa array identik menghasilkan MSE = 0, PSNR = inf, SSIM = 1.0, BER = 0.
   - Verifikasi deteksi perubahan bit buatan untuk menguji akurasi BER.

---

### **Fase 2: Milestone 1 — Baseline LSB Konvensional**
> **Fokus:** Mengimplementasikan penyisipan domain spasial LSB murni sebagai baseline perbandingan (FR-03, FR-04).

1. **Implementasi Algoritma (`src/lsb.py`):**
   - Skema embedding sekuensial pada bit ke-0 (LSB) piksel citra.
   - Penambahan header protokol (32-bit integer awal untuk panjang pesan) agar proses ekstraksi bersifat *blind* (tidak memerlukan citra asli).
   - Fungsi `embed(cover_image, payload_bytes)` $\to$ `stego_image, time_sec, info`.
   - Fungsi `extract(stego_image)` $\to$ `recovered_bytes, time_sec, info`.

2. **Validasi Skenario A (Baseline tanpa Serangan):**
   - Memastikan pesan sintetis berhasil diekstrak 100% utuh ($BER = 0$, $ESR = 100\%$).
   - Mencatat nilai PSNR awal ($\ge 50$ dB pada payload wajar).
   - Unit test pada `tests/test_lsb.py`.

---

### **Fase 3: Milestone 2 — Content-Adaptive LSB**
> **Fokus:** Mengimplementasikan penyisipan spasial adaptif berdasarkan karakteristik tekstur dan visual citra (FR-05).

1. **Implementasi Algoritma (`src/adaptive_lsb.py`):**
   - Analisis kompleksitas konten lokal citra (menggunakan operator gradien seperti Sobel, varians lokal, atau visual saliency map).
   - Memisahkan area citra menjadi area homogen/halus vs area bertekstur tinggi/tulang.
   - Peta prioritas penyisipan: memprioritaskan piksel pada area bertekstur tinggi agar noise steganografi tersamarkan oleh tekstur alami citra.
   - Mekanisme rekonstruksi lokasi ekstraksi (menggunakan koordinat deterministik berbasis seed pseudo-random atau metadata lokasi).
   - Fungsi `embed` dan `extract` dengan struktur output standar.

2. **Validasi Skenario A:**
   - Memverifikasi keberhasilan ekstraksi baseline tanpa serangan.
   - Membandingkan PSNR dan SSIM terhadap LSB biasa pada payload yang sama.
   - Unit test pada `tests/test_adaptive_lsb.py`.

---

### **Fase 4: Milestone 3 — DCT-Based Steganography**
> **Fokus:** Mengimplementasikan steganografi domain frekuensi DCT untuk mengevaluasi ketahanan terhadap kompresi (FR-06).

1. **Implementasi Algoritma (`src/dct.py`):**
   - Partisi citra ke dalam blok $8 \times 8$ piksel.
   - Transformasi 2D DCT (`scipy.fftpack.dct`) pada setiap blok.
   - Pemilihan koefisien frekuensi menengah (*mid-frequency coefficients*): menghindari frekuensi tinggi yang terbuang saat kuantisasi JPEG dan frekuensi rendah (DC) yang mempengaruhi kontras visual.
   - Aturan penyisipan bit (misal: modifikasi selisih pasangan koefisien kuantisasi).
   - Inverse 2D DCT (`idct`), pembulatan (*rounding*), dan pemotongan (*clipping*) piksel ke rentang `[0, 255]`.
   - Prosedur ekstraksi bit dari koefisien frekuensi blok.

2. **Validasi Skenario A:**
   - Memeriksa batas kapasitas payload blok $8 \times 8$.
   - Memvalidasi ekstraksi tanpa serangan dan mengukur error akibat proses pembulatan (clipping effect).
   - Unit test pada `tests/test_dct.py`.

---

### **Fase 5: Milestone 4 — Simulasi Serangan Kompresi JPEG**
> **Fokus:** Menguji ketahanan ketiga metode terhadap kompresi JPEG dengan berbagai faktor kualitas (FR-05, FR-07).

1. **Modul Simulasi JPEG (`src/jpeg_attack.py`):**
   - Menyimpan stego-image ke format JPEG dengan faktor kualitas $Q \in \{95, 90, 85, 80, 75\}$ melalui Pillow/OpenCV.
   - Membaca kembali file terkompresi sebagai citra uji terdegradasi.
   - Menyimpan seluruh file uji di direktori `outputs/compressed/` tanpa menimpa stego-image asli.

2. **Penanganan Kegagalan Ekstraksi Graceful (FR-04):**
   - Mengantisipasi kerusakan header akibat kompresi (khususnya pada LSB spasial).
   - Sistem mencatat kegagalan secara eksplisit (`extraction_success = False`, nilai BER dihitung dari bit yang terdeteksi atau diisi penanda kegagalan valid), tanpa membuat aplikasi crash.

---

### **Fase 6: Otomatisasi Orkestrasi Eksperimen & Export Data**
> **Fokus:** Mengintegrasikan seluruh pipeline ke dalam satu perintah otomatis dan mengekspor log eksperimen ke CSV (FR-07, FR-11, FR-14, FR-15).

1. **Modul Orkestrator (`src/experiment.py`):**
   - Menjalankan iterasi otomatis:
     $$\text{Dataset Citra} \times \text{Metode (LSB, Adaptive, DCT)} \times \text{Ukuran Payload} \times \text{Kualitas JPEG}$$
   - Melakukan 3 pasang komparasi metrik kualitas:
     1. **Cover vs Stego** (distorsi embedding awal).
     2. **Stego vs Stego Compressed** (distorsi akibat kompresi).
     3. **Cover vs Stego Compressed** (distorsi total).
   - Mengukur waktu eksekusi: *Embedding Time* dan *Extraction Time*.

2. **Logging Tabular CSV (`results/experiment_results.csv`):**
   Struktur 16 kolom wajib sesuai PRD:
   - `experiment_id`, `image_id`, `method`, `payload_bytes`, `payload_bpp`, `jpeg_quality`
   - `mse_cover_stego`, `psnr_cover_stego`, `ssim_cover_stego`
   - `mse_stego_compressed`, `psnr_stego_compressed`, `ssim_stego_compressed`
   - `ber`, `extraction_success`, `embedding_time_seconds`, `extraction_time_seconds`

---

### **Fase 7: Milestone 5 — Visualisasi, Analisis, & Pelaporan**
> **Fokus:** Mengubah data CSV menjadi visualisasi grafik dan tabel komparatif siap pakai (FR-12, FR-13).

1. **Modul Pelaporan (`src/reporting.py`):**
   - **Grafik Kualitas Citra vs Payload:** Kurva PSNR & SSIM terhadap variasi ukuran payload untuk ketiga metode.
   - **Grafik Ketahanan vs Kualitas JPEG:** Kurva BER & Extraction Success Rate terhadap variasi kualitas JPEG ($95 \to 75$).
   - **Grafik Waktu Komputasi:** Perbandingan efisiensi waktu embedding dan ekstraksi antarmetode.
   - **Tabel Ringkasan:** Rekapitulasi rata-rata performa dan trade-off antar ketiga metode.
   - Menyimpan seluruh grafik ke `results/figures/` dalam resolusi publikasi (300 DPI).

---

### **Fase 8: Validasi Kriteria Keberhasilan & Finalisasi**
> **Fokus:** Verifikasi menyeluruh terhadap Definition of Done dan kriteria penerimaan (Bab 10 & 16 PRD).

- [ ] **AC-01:** Pembacaan dataset citra X-ray valid tanpa korupsi.
- [ ] **AC-02:** LSB berhasil embed & extract tanpa serangan ($BER = 0$).
- [ ] **AC-03:** Content-Adaptive LSB berhasil embed & extract tanpa serangan ($BER = 0$).
- [ ] **AC-04:** DCT berhasil embed & extract tanpa serangan.
- [ ] **AC-05:** Simulasi JPEG berjalan lancar untuk kualitas 95, 90, 85, 80, dan 75.
- [ ] **AC-06:** Penghitungan MSE, PSNR, SSIM, BER, ESR akurat dan terdokumentasi.
- [ ] **AC-07:** Penanganan extraction failure tercatat eksplisit pada data eksperimen.
- [ ] **AC-08 & AC-09:** File `results/experiment_results.csv` terisi lengkap dengan seluruh parameter.
- [ ] **AC-10:** Eksperimen dapat diulang dengan hasil yang identik (*reproducible*).
- [ ] **AC-11:** Ringkasan tabel dan visualisasi grafik perbandingan ter-generate.
- [ ] **AC-12:** Citra sumber tidak tertimpa/termodifikasi selama proses pengujian.
- [ ] Pembuatan `README.md` komprehensif berisi petunjuk instalasi dan eksekusi eksperimen.

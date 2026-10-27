# X-Ray Steganography Research Platform

Platform penelitian eksperimental komparatif untuk mengevaluasi tiga metode steganografi citra pada citra medis X-Ray dada:
1. **Least Significant Bit (LSB)** - Domain Spasial Konvensional
2. **Content-Adaptive LSB** - Domain Spasial Adaptif (Tekstur/Saliency)
3. **Discrete Cosine Transform (DCT)** - Domain Frekuensi

Proyek ini mengevaluasi trade-off antara **kualitas visual citra** (MSE, PSNR, SSIM) dan **ketahanan ekstraksi pesan** (BER, Extraction Success Rate) terhadap variasi ukuran payload serta serangan kompresi JPEG ($Q = 95, 90, 85, 80, 75$).

---

## 📁 Struktur Direktori

```text
steganography/
├── config.yaml              # Konfigurasi parameter eksperimen terpusat
├── requirements.txt         # Daftar dependensi library Python
├── README.md                # Dokumentasi petunjuk penggunaan
├── docs/
│   ├── PRD.md               # Product Requirements Document
│   └── STEP.md              # Rencana tahapan pengerjaan (step-by-step)
├── data/
│   ├── cover/               # Citra referensi X-Ray dada (PNG grayscale 8-bit)
│   └── secret/              # Pesan rahasia sintetis pasien (UTF-8)
├── outputs/
│   ├── stego/               # Citra stego hasil penyisipan
│   ├── compressed/          # Citra stego setelah kompresi JPEG
│   └── recovered/           # Teks pesan hasil ekstraksi
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Validasi citra & pemuat pesan
│   ├── metrics.py           # Perhitungan MSE, PSNR, SSIM, BER, ESR, BPP
│   ├── lsb.py               # Algoritma LSB konvensional
│   ├── adaptive_lsb.py      # Algoritma Content-Adaptive LSB
│   ├── dct.py               # Algoritma DCT domain frekuensi
│   ├── jpeg_attack.py       # Simulasi kompresi JPEG
│   ├── experiment.py        # Orkestrator eksperimen batch
│   └── reporting.py         # Visualisasi grafik dan tabel komparasi
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_lsb.py
│   ├── test_adaptive_lsb.py
│   └── test_dct.py
└── results/
    ├── experiment_results.csv # Log tabular 16 kolom metrik eksperimen
    └── figures/             # Grafik visualisasi analitis (300 DPI)
```

---

## 🚀 Instalasi & Persiapan Lingkungan

1. **Prasyarat:**
   - Python 3.10 atau versi lebih baru (diuji pada Python 3.13)

2. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Konfigurasi (`config.yaml`)

Seluruh parameter eksperimen dikontrol melalui `config.yaml`:
- **Payload:** `64, 128, 256, 512, 1024` byte.
- **Faktor Kualitas JPEG:** `95, 90, 85, 80, 75`.
- **Parameter DCT:** Ukuran blok $8 \times 8$, pasangan koefisien frekuensi menengah $(4, 3)$ dan $(3, 4)$.
- **Reproducibility:** Seed acak `42`.

---

## 🧪 Menjalankan Pengujian

Jalankan rangkaian unit testing menggunakan `pytest`:
```bash
pytest tests/
```

---

## 📜 Catatan Integritas Data & Etika Penelitian
- Platform ini merupakan **prototipe penelitian** dan tidak digunakan untuk diagnosis klinis.
- Menggunakan data pasien sintetis untuk melindungi privasi (*NFR-05*).
- Citra referensi asli pada `data/cover/` bersifat *read-only* dan tidak akan ditimpa selama pengujian (*NFR-02*).

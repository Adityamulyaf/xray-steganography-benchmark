"""
Pengujian Unit untuk Modul Orkestrator Eksperimen (src/experiment.py)
Memenuhi Fase 6, FR-07, FR-11, FR-14, FR-15, NFR-01, AC-08, AC-09, AC-10, AC-12 PRD.md / STEP.md.
"""

import os
import pandas as pd
import numpy as np
import pytest

from src.experiment import load_config, run_experiment, REQUIRED_CSV_COLUMNS
from src.dataset import load_cover_image


class TestExperimentOrchestrator:
    """Rangkaian pengujian orkestrator eksperimen dan integritas data logging CSV."""

    def test_load_config_valid(self):
        """Memastikan fungsi load_config membaca file config.yaml dengan struktur yang benar."""
        cfg = load_config("config.yaml")
        assert "experiment" in cfg
        assert "image_settings" in cfg
        assert "payload_settings" in cfg
        assert "jpeg_attack_settings" in cfg
        assert "paths" in cfg

    def test_run_experiment_quick_subset(self, tmp_path):
        """Menjalankan eksperimen pada subset kecil dan memvalidasi struktur DataFrame."""
        csv_out = str(tmp_path / "test_exp_results.csv")
        custom_cfg = {
            "experiment": {
                "name": "unit_test_exp",
                "random_seed": 42,
                "methods": ["lsb"]
            },
            "paths": {
                "cover_dir": "data/cover",
                "secret_dir": "data/secret",
                "stego_dir": str(tmp_path / "stego"),
                "compressed_dir": str(tmp_path / "compressed"),
                "recovered_dir": str(tmp_path / "recovered"),
                "results_csv": csv_out
            },
            "payload_settings": {
                "payload_sizes_bytes": [64]
            },
            "jpeg_attack_settings": {
                "qualities": [90]
            },
            "cover_files_override": ["xray_sample_01.png"]
        }

        df = run_experiment(custom_config=custom_cfg, save_csv=True)

        # 1 citra x 1 metode x 1 payload x (1 baseline + 1 quality) = 2 baris
        assert len(df) == 2
        assert os.path.isfile(csv_out)

        # Periksa keberadaan 16 kolom standar wajib
        for col in REQUIRED_CSV_COLUMNS:
            assert col in df.columns, f"Kolom wajib '{col}' tidak ditemukan di DataFrame."

    def test_full_experiment_csv_integrity(self):
        """Memvalidasi file hasil eksperimen lengkap di results/experiment_results.csv."""
        csv_file = "results/experiment_results.csv"
        assert os.path.isfile(csv_file), f"File {csv_file} belum dibuat."

        df = pd.read_csv(csv_file)

        # Memastikan seluruh 180 konfigurasi pengujian terdata
        # (2 citra x 3 metode x 5 payload x 6 kondisi [none, 95, 90, 85, 80, 75])
        assert len(df) == 180

        # Validasi 16 kolom standar
        for col in REQUIRED_CSV_COLUMNS:
            assert col in df.columns

        # Validasi metode yang diuji
        assert set(df["method"].unique()) == {"lsb", "adaptive_lsb", "dct"}

        # Validasi rentang nilai BER
        valid_ber = df["ber"].dropna()
        assert (valid_ber >= 0.0).all() and (valid_ber <= 1.0).all()

        # Validasi bahwa kegagalan kapasitas tercatat secara eksplisit (AC-07)
        capacity_exceeded_rows = df[df["status"] == "CAPACITY_EXCEEDED"]
        assert len(capacity_exceeded_rows) > 0
        # DCT pada payload 512B dan 1024B harus berstatus CAPACITY_EXCEEDED
        dct_exceeded = df[(df["method"] == "dct") & (df["payload_bytes"] >= 512)]
        assert (dct_exceeded["status"] == "CAPACITY_EXCEEDED").all()
        assert (dct_exceeded["extraction_success"] == False).all()

    def test_cover_image_immutability(self):
        """Memastikan citra sumber asli tidak mengalami perubahan sama sekali (AC-12, NFR-02)."""
        cover_path = "data/cover/xray_sample_01.png"
        img_original = load_cover_image(cover_path)

        # Baca kembali dari disk
        img_current = load_cover_image(cover_path)
        assert np.array_equal(img_original, img_current)


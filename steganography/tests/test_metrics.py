"""
Pengujian Unit untuk Modul Metrik Evaluasi (src/metrics.py)
Memenuhi Persyaratan Pengujian FR-06, FR-08, FR-09 pada STEP.md Fase 1.
"""

import math
import numpy as np
import pytest

from src.metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    calculate_bpp,
    calculate_ber,
    calculate_image_metrics,
    calculate_message_metrics
)
from src.dataset import bytes_to_bits


class TestImageMetrics:
    """Pengujian metrik kualitas citra: MSE, PSNR, SSIM."""

    def test_identical_images(self):
        """Dua citra identik harus menghasilkan MSE=0, PSNR=inf, dan SSIM=1.0."""
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        mse = calculate_mse(img, img)
        psnr = calculate_psnr(img, img)
        ssim = calculate_ssim(img, img)
        
        assert mse == 0.0
        assert math.isinf(psnr)
        assert pytest.approx(ssim, rel=1e-5) == 1.0

    def test_known_mse_and_psnr(self):
        """Memverifikasi perhitungan MSE dan PSNR dengan deviasi piksel konstan."""
        img1 = np.full((10, 10), 100, dtype=np.uint8)
        # Selisih 1 pada tiap piksel -> MSE = 1.0, PSNR = 10 * log10(255^2 / 1) ≈ 48.1308 dB
        img2 = np.full((10, 10), 101, dtype=np.uint8)
        
        mse = calculate_mse(img1, img2)
        psnr = calculate_psnr(img1, img2)
        
        assert pytest.approx(mse, rel=1e-5) == 1.0
        expected_psnr = 10.0 * math.log10(255.0 ** 2 / 1.0)
        assert pytest.approx(psnr, rel=1e-4) == expected_psnr

    def test_ssim_degradation(self):
        """SSIM harus menurun saat noise ditambahkan."""
        img = np.ones((128, 128), dtype=np.uint8) * 128
        noisy_img = np.clip(img.astype(np.int16) + np.random.randint(-20, 20, (128, 128)), 0, 255).astype(np.uint8)
        
        ssim_val = calculate_ssim(img, noisy_img)
        assert 0.0 <= ssim_val < 1.0

    def test_dimension_mismatch_raises_error(self):
        """Menghitung MSE atau SSIM dengan dimensi berbeda harus memicu ValueError."""
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.zeros((100, 120), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            calculate_mse(img1, img2)
            
        with pytest.raises(ValueError):
            calculate_ssim(img1, img2)

    def test_calculate_image_metrics_dict(self):
        """Fungsi pembungkus calculate_image_metrics harus mengembalikan kamus lengkap."""
        img1 = np.full((64, 64), 50, dtype=np.uint8)
        img2 = np.full((64, 64), 52, dtype=np.uint8)
        
        metrics = calculate_image_metrics(img1, img2)
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert metrics["mse"] == 4.0


class TestMessageMetrics:
    """Pengujian metrik ekstraksi pesan: BER, ESR, dan BPP."""

    def test_identical_message_metrics(self):
        """Pesan yang identik 100% harus menghasilkan BER=0 dan extraction_success=True."""
        original = b"PATIENT_DATA_SECRET_123"
        recovered = b"PATIENT_DATA_SECRET_123"
        
        result = calculate_message_metrics(original, recovered)
        assert result["ber"] == 0.0
        assert result["extraction_success"] is True

    def test_controlled_bit_errors(self):
        """Memverifikasi BER jika ada tepat 1 bit yang salah dari 8 bit (1 byte)."""
        # Byte 'A' = 01000001
        # Byte 'C' = 01000011 -> beda 1 bit dari 'A'
        orig = b"A"  # 8 bit
        rec = b"C"   # 8 bit (beda 1 bit)
        
        ber = calculate_ber(orig, rec)
        # 1 bit salah dari 8 bit = 1/8 = 0.125
        assert pytest.approx(ber, rel=1e-5) == 0.125
        
        metrics = calculate_message_metrics(orig, rec)
        assert metrics["ber"] == 0.125
        assert metrics["extraction_success"] is False

    def test_total_failure_metrics(self):
        """Jika ekstraksi gagal total (None), BER harus 1.0 dan extraction_success False."""
        orig = b"IMPORTANT_PATIENT_METADATA"
        result = calculate_message_metrics(orig, None)
        
        assert result["ber"] == 1.0
        assert result["extraction_success"] is False

    def test_length_mismatch_penalty(self):
        """Pesan terekstrak yang terpotong harus mendapat penalti BER yang proporsional."""
        orig = b"ABCDEFGH"  # 8 byte = 64 bit
        rec = b"ABCD"       # 4 byte = 32 bit (32 bit hilang)
        
        ber = calculate_ber(orig, rec)
        # 32 bit hilang dari 64 bit = 32/64 = 0.5
        assert pytest.approx(ber, rel=1e-5) == 0.5

    def test_bpp_calculation(self):
        """Verifikasi penghitungan Bits Per Pixel (BPP)."""
        payload_bytes = 1024  # 1024 * 8 = 8192 bit
        image_shape = (512, 512)  # 262144 piksel
        
        bpp = calculate_bpp(payload_bytes, image_shape)
        expected_bpp = 8192.0 / 262144.0  # 0.03125
        assert pytest.approx(bpp, rel=1e-5) == expected_bpp

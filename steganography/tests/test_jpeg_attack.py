"""
Pengujian Unit untuk Modul Simulasi Serangan Kompresi JPEG (src/jpeg_attack.py)
Memenuhi Fase 5, FR-05, FR-07, FR-09, dan AC-05 PRD.md / STEP.md.
"""

import os
import numpy as np
import pytest

from src.dataset import load_cover_image, load_secret_message
from src.lsb import embed_lsb
from src.adaptive_lsb import embed_adaptive_lsb
from src.dct import embed_dct
from src.jpeg_attack import compress_jpeg, evaluate_jpeg_attack, DEFAULT_JPEG_QUALITIES


class TestJPEGAttack:
    """Rangkaian pengujian simulasi serangan JPEG dan evaluasi degradasi citra."""

    @pytest.fixture
    def sample_cover(self) -> np.ndarray:
        """Memuat citra X-Ray sampel 512x512."""
        return load_cover_image("data/cover/xray_sample_01.png")

    @pytest.fixture
    def sample_secret(self) -> bytes:
        """Memuat pesan rahasia 64 byte."""
        return load_secret_message("data/secret/sample_64bytes.txt")

    def test_compress_jpeg_valid_outputs(self, sample_cover, tmp_path):
        """Memastikan fungsi kompresi menghasilkan citra berukuran sama dan file valid."""
        out_path = str(tmp_path / "test_q90.jpg")
        decomp, size_bytes = compress_jpeg(sample_cover, quality=90, output_path=out_path)

        assert decomp.shape == sample_cover.shape
        assert decomp.dtype == np.uint8
        assert size_bytes > 0
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) == size_bytes

    def test_compression_ratio_trend(self, sample_cover):
        """Memastikan bahwa ukuran stream JPEG menurun seiring penurunan faktor kualitas Q."""
        _, size_95 = compress_jpeg(sample_cover, quality=95)
        _, size_85 = compress_jpeg(sample_cover, quality=85)
        _, size_75 = compress_jpeg(sample_cover, quality=75)

        assert size_95 > size_85 > size_75

    def test_invalid_quality_raises_error(self, sample_cover):
        """Kualitas di luar rentang 1-100 harus menghasilkan ValueError."""
        with pytest.raises(ValueError, match="1-100"):
            compress_jpeg(sample_cover, quality=0)

        with pytest.raises(ValueError, match="1-100"):
            compress_jpeg(sample_cover, quality=101)

    def test_lsb_fails_gracefully_under_jpeg(self, sample_cover, sample_secret):
        """
        LSB konvensional pada domain spasial harus mencatat kegagalan secara eksplisit
        (BER tinggi, ESR=False) tanpa membuat aplikasi crash (FR-04).
        """
        stego_lsb, _, _ = embed_lsb(sample_cover, sample_secret)
        res = evaluate_jpeg_attack(
            cover_image=sample_cover,
            stego_image=stego_lsb,
            method="lsb",
            original_payload=sample_secret,
            quality=90
        )

        assert res["extraction_success"] is False
        assert res["ber"] > 0.30
        assert "mse_cover_stego" in res
        assert "psnr_stego_compressed" in res

    def test_adaptive_lsb_fails_gracefully_under_jpeg(self, sample_cover, sample_secret):
        """Content-Adaptive LSB spasial juga harus mencatat kegagalan secara eksplisit tanpa crash."""
        stego_adapt, _, _ = embed_adaptive_lsb(sample_cover, sample_secret)
        res = evaluate_jpeg_attack(
            cover_image=sample_cover,
            stego_image=stego_adapt,
            method="adaptive_lsb",
            original_payload=sample_secret,
            quality=90
        )

        assert res["extraction_success"] is False
        assert res["ber"] > 0.30

    def test_dct_robustness_at_high_quality_jpeg(self, sample_cover, sample_secret):
        """
        Steganografi domain frekuensi DCT harus tahan terhadap kompresi JPEG kualitas tinggi
        (Q=95 dan Q=90) dengan BER=0 dan ESR=True.
        """
        stego_dct, _, _ = embed_dct(sample_cover, sample_secret, quantization_factor=25.0)

        # Uji kualitas 95
        res_95 = evaluate_jpeg_attack(
            cover_image=sample_cover,
            stego_image=stego_dct,
            method="dct",
            original_payload=sample_secret,
            quality=95
        )
        assert res_95["extraction_success"] is True
        assert res_95["ber"] == 0.0

        # Uji kualitas 90
        res_90 = evaluate_jpeg_attack(
            cover_image=sample_cover,
            stego_image=stego_dct,
            method="dct",
            original_payload=sample_secret,
            quality=90
        )
        assert res_90["extraction_success"] is True
        assert res_90["ber"] == 0.0

    def test_image_immutability_under_attack(self, sample_cover, sample_secret, tmp_path):
        """Memastikan citra cover asli dan stego asli tidak termutasi selama simulasi serangan (NFR-02)."""
        cover_copy = sample_cover.copy()
        stego_dct, _, _ = embed_dct(sample_cover, sample_secret)
        stego_copy = stego_dct.copy()

        out_file = str(tmp_path / "compressed_test.jpg")
        _ = evaluate_jpeg_attack(
            cover_image=sample_cover,
            stego_image=stego_dct,
            method="dct",
            original_payload=sample_secret,
            quality=85,
            compressed_output_path=out_file
        )

        assert np.array_equal(sample_cover, cover_copy)
        assert np.array_equal(stego_dct, stego_copy)


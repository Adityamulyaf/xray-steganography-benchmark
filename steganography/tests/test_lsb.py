"""
Pengujian Unit untuk Algoritma LSB Konvensional (src/lsb.py)
Memenuhi Skenario A (Baseline tanpa Serangan), FR-03, FR-04, dan AC-02.
"""

import numpy as np
import pytest

from src.lsb import embed_lsb, extract_lsb
from src.dataset import load_cover_image, load_secret_message
from src.metrics import calculate_image_metrics, calculate_message_metrics


class TestLSBSteganography:
    """Rangkaian pengujian embedding dan ekstraksi LSB konvensional."""

    @pytest.fixture
    def sample_cover(self) -> np.ndarray:
        """Memuat citra X-Ray sampel 512x512."""
        return load_cover_image("data/cover/xray_sample_01.png")

    def test_baseline_extraction_exact_match(self, sample_cover):
        """Memastikan data pulih 100% utuh tanpa serangan (BER=0, ESR=True)."""
        secret_payload = b"PATIENT_ID:PX-001|NAME:JOHN DOE|STATUS:CONFIRMED"
        
        stego_img, embed_time, embed_info = embed_lsb(sample_cover, secret_payload)
        assert embed_info["success"] is True
        assert embed_time > 0.0
        
        recovered_payload, extract_time, extract_info = extract_lsb(stego_img)
        assert extract_info["success"] is True
        assert extract_time > 0.0
        assert recovered_payload == secret_payload
        
        # Metrik pesan harus sempurna
        msg_metrics = calculate_message_metrics(secret_payload, recovered_payload)
        assert msg_metrics["ber"] == 0.0
        assert msg_metrics["extraction_success"] is True

    @pytest.mark.parametrize("payload_size", [64, 128, 256, 512, 1024])
    def test_baseline_various_payload_sizes(self, sample_cover, payload_size):
        """Menguji embedding dan ekstraksi untuk seluruh variasi ukuran payload target."""
        secret_file = f"data/secret/sample_{payload_size}bytes.txt"
        payload = load_secret_message(secret_file)
        assert len(payload) == payload_size
        
        stego_img, _, _ = embed_lsb(sample_cover, payload)
        recovered, _, info = extract_lsb(stego_img)
        
        assert info["success"] is True
        assert recovered == payload
        
        # Kualitas citra stego harus sangat tinggi (PSNR > 50 dB)
        img_metrics = calculate_image_metrics(sample_cover, stego_img)
        assert img_metrics["psnr"] >= 50.0
        assert img_metrics["ssim"] > 0.999

    def test_cover_image_immutability(self, sample_cover):
        """Memastikan citra sumber referensi tidak termodifikasi selama embedding (NFR-02)."""
        original_copy = sample_cover.copy()
        payload = b"TEST_IMMUTABILITY_SECRET"
        
        stego_img, _, _ = embed_lsb(sample_cover, payload)
        
        # Citra cover asli harus tetap sama persis dengan sebelum embedding
        assert np.array_equal(sample_cover, original_copy)
        # Citra stego harus berbeda dari citra cover
        assert not np.array_equal(sample_cover, stego_img)

    def test_exceeding_capacity_raises_error(self):
        """Penyisipan yang melebihi batas kapasitas citra harus ditolak dengan ValueError."""
        small_cover = np.zeros((10, 10), dtype=np.uint8)  # 100 piksel = 100 bit
        # 20 bytes = 160 bit payload + 32 bit header = 192 bit > 100 bit kapasitas
        excessive_payload = b"A" * 20
        
        with pytest.raises(ValueError, match="melebihi kapasitas"):
            embed_lsb(small_cover, excessive_payload)

    def test_corrupted_header_handling(self, sample_cover):
        """Jika header rusak atau mengindikasikan payload mustahil, ekstraksi harus gagal aman."""
        stego_img, _, _ = embed_lsb(sample_cover, b"VALID_PAYLOAD")
        
        # Rusak 4 byte pertama (header) menjadi nilai raksasa
        corrupted_stego = stego_img.copy()
        corrupted_stego[0, :32] = 1  # 32 bit bernilai 1 -> int 4.294.967.295 bytes
        
        recovered, _, info = extract_lsb(corrupted_stego)
        assert recovered is None
        assert info["success"] is False
        assert "melebihi ukuran citra" in info["error"]

    def test_empty_payload(self, sample_cover):
        """Penyisipan payload kosong (0 bytes) harus berhasil ditangani."""
        stego_img, _, _ = embed_lsb(sample_cover, b"")
        recovered, _, info = extract_lsb(stego_img)
        
        assert info["success"] is True
        assert recovered == b""

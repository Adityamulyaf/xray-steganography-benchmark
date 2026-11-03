"""
Pengujian Unit untuk Algoritma DCT Steganography (src/dct.py)
Memenuhi Fase 4, Skenario A (Baseline tanpa Serangan), FR-06, dan AC-04 PRD.md / STEP.md.
"""

import numpy as np
import pytest

from src.dct import embed_dct, extract_dct, dct2, idct2
from src.dataset import load_cover_image, load_secret_message
from src.metrics import calculate_image_metrics, calculate_message_metrics


class TestDCTSteganography:
    """Rangkaian pengujian embedding dan ekstraksi steganografi domain frekuensi DCT."""

    @pytest.fixture
    def sample_cover(self) -> np.ndarray:
        """Memuat citra X-Ray sampel 512x512."""
        return load_cover_image("data/cover/xray_sample_01.png")

    def test_baseline_extraction_exact_match(self, sample_cover):
        """Memastikan data pulih 100% utuh tanpa serangan pada domain frekuensi DCT (BER=0, ESR=True)."""
        secret_payload = b"PATIENT_ID:PX-DCT-001|DIAGNOSIS:NORMAL|CHEST_XRAY"

        stego_img, embed_time, embed_info = embed_dct(sample_cover, secret_payload)
        assert embed_info["success"] is True
        assert embed_time > 0.0

        recovered_payload, extract_time, extract_info = extract_dct(stego_img)
        assert extract_info["success"] is True
        assert extract_time > 0.0
        assert recovered_payload == secret_payload

        # Metrik pesan harus sempurna tanpa serangan
        msg_metrics = calculate_message_metrics(secret_payload, recovered_payload)
        assert msg_metrics["ber"] == 0.0
        assert msg_metrics["extraction_success"] is True

    @pytest.mark.parametrize("payload_size", [64, 128, 256])
    def test_baseline_various_payload_sizes(self, sample_cover, payload_size):
        """Menguji embedding dan ekstraksi untuk payload yang muat dalam kapasitas blok 8x8 citra 512x512."""
        secret_file = f"data/secret/sample_{payload_size}bytes.txt"
        payload = load_secret_message(secret_file)
        assert len(payload) == payload_size

        stego_img, _, embed_info = embed_dct(sample_cover, payload)
        assert embed_info["success"] is True

        recovered, _, extract_info = extract_dct(stego_img)
        assert extract_info["success"] is True
        assert recovered == payload

        # Kualitas citra stego DCT tetap tinggi (PSNR > 40 dB, SSIM > 0.95)
        img_metrics = calculate_image_metrics(sample_cover, stego_img)
        assert img_metrics["psnr"] >= 40.0
        assert img_metrics["ssim"] >= 0.95

    def test_cover_image_immutability(self, sample_cover):
        """Memastikan citra sumber referensi tidak termodifikasi selama embedding (NFR-02)."""
        original_copy = sample_cover.copy()
        payload = b"TEST_DCT_IMMUTABILITY_SECRET"

        stego_img, _, _ = embed_dct(sample_cover, payload)

        # Citra cover asli harus tetap sama persis dengan sebelum embedding
        assert np.array_equal(sample_cover, original_copy)
        # Citra stego harus berbeda dari citra cover
        assert not np.array_equal(sample_cover, stego_img)

    def test_dct_roundtrip_orthonormal(self):
        """Memastikan bahwa 2D DCT dan 2D IDCT orthonormal adalah inverse sempurna."""
        block = np.random.uniform(0, 255, (8, 8))
        freq = dct2(block)
        reconstructed = idct2(freq)
        assert np.allclose(block, reconstructed, atol=1e-5)

    def test_exceeding_capacity_raises_error(self, sample_cover):
        """
        Pada citra 512x512 (4096 blok 8x8), kapasitas maksimum adalah 508 bytes payload + 4 bytes header.
        Payload 512 bytes dan 1024 bytes harus ditolak dengan ValueError.
        """
        excessive_payload_512 = b"A" * 512
        with pytest.raises(ValueError, match="melebihi kapasitas"):
            embed_dct(sample_cover, excessive_payload_512)

        excessive_payload_1024 = b"A" * 1024
        with pytest.raises(ValueError, match="melebihi kapasitas"):
            embed_dct(sample_cover, excessive_payload_1024)

    def test_invalid_dimension_raises_error(self):
        """Citra dengan dimensi yang tidak habis dibagi 8 harus menghasilkan ValueError."""
        invalid_img = np.zeros((100, 100), dtype=np.uint8)  # 100 bukan kelipatan 8
        with pytest.raises(ValueError, match="habis dibagi block_size"):
            embed_dct(invalid_img, b"TEST")

    def test_corrupted_header_handling(self, sample_cover):
        """Jika header DCT rusak dan mengindikasikan payload mustahil, ekstraksi harus gagal aman."""
        stego_img, _, _ = embed_dct(sample_cover, b"VALID_DCT_PAYLOAD")

        # Modifikasi koefisien pada blok-blok header menjadi kondisi ekstrem
        corrupted_stego = stego_img.copy()
        # Modifikasi 32 blok pertama dengan nilai random ekstrem
        corrupted_stego[:8, :256] = np.random.randint(0, 256, (8, 256), dtype=np.uint8)

        recovered, _, info = extract_dct(corrupted_stego)
        # Ekstraksi harus gagal secara graceful (success = False) tanpa crash
        assert (recovered is None) or (info["success"] is False)

    def test_empty_payload(self, sample_cover):
        """Penyisipan payload kosong (0 bytes) pada DCT harus ditangani secara aman."""
        stego_img, _, _ = embed_dct(sample_cover, b"")
        recovered, _, info = extract_dct(stego_img)

        assert info["success"] is True
        assert recovered == b""


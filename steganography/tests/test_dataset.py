"""
Pengujian Unit untuk Modul Dataset (src/dataset.py)
Memverifikasi I/O Citra, Konversi Biner, dan Pemeriksaan Kapasitas.
"""

import os
import numpy as np
import pytest

from src.dataset import (
    load_cover_image,
    save_image,
    load_secret_message,
    bytes_to_bits,
    bits_to_bytes,
    check_capacity
)


class TestDatasetModule:
    """Pengujian operasi dataset citra dan bit stream."""

    def test_load_valid_cover_image(self):
        """Memastikan citra sample cover dapat dimuat dengan shape dan dtype yang benar."""
        img_path = "data/cover/xray_sample_01.png"
        img = load_cover_image(img_path)
        
        assert isinstance(img, np.ndarray)
        assert img.shape == (512, 512)
        assert img.dtype == np.uint8

    def test_load_nonexistent_image_raises_error(self):
        """Memastikan FileNotFoundError dimunculkan jika file tidak ditemukan."""
        with pytest.raises(FileNotFoundError):
            load_cover_image("data/cover/non_existent_file.png")

    def test_bytes_and_bits_roundtrip(self):
        """Memastikan bytes -> bits -> bytes menghasilkan data yang identik 100%."""
        test_data = b"Hello, X-Ray Steganography! 12345 @#$%"
        bits = bytes_to_bits(test_data)
        
        # Panjang bit harus len(data) * 8
        assert len(bits) == len(test_data) * 8
        # Seluruh elemen harus berupa 0 atau 1
        assert np.all((bits == 0) | (bits == 1))
        
        reconstructed = bits_to_bytes(bits)
        assert reconstructed == test_data

    def test_empty_bytes_to_bits(self):
        """Memastikan data bytes kosong ditangani dengan aman."""
        bits = bytes_to_bits(b"")
        assert len(bits) == 0
        assert bits_to_bytes(bits) == b""

    def test_invalid_bits_length_raises_error(self):
        """Array bit dengan panjang bukan kelipatan 8 harus memicu ValueError."""
        invalid_bits = np.array([1, 0, 1], dtype=np.uint8)
        with pytest.raises(ValueError):
            bits_to_bytes(invalid_bits)

    def test_check_capacity_lsb(self):
        """Pengujian kapasitas untuk metode LSB."""
        shape = (100, 100)  # 10.000 piksel = 10.000 bit kapasitas
        # Payload 100 bytes + 4 bytes overhead = 104 bytes = 832 bit -> CUKUP
        sufficient, req, max_cap = check_capacity(shape, 100, method="lsb")
        assert sufficient is True
        assert req == 832
        assert max_cap == 10000
        
        # Payload 2000 bytes = 16032 bit -> TIDAK CUKUP
        sufficient_over, req_over, _ = check_capacity(shape, 2000, method="lsb")
        assert sufficient_over is False
        assert req_over > max_cap

    def test_check_capacity_dct(self):
        """Pengujian kapasitas untuk metode DCT (1 bit per blok 8x8)."""
        shape = (512, 512)
        # 512 / 8 = 64 blok per baris/kolom -> 64 * 64 = 4096 blok = 4096 bit kapasitas
        sufficient, req, max_cap = check_capacity(shape, 256, method="dct")
        # 256 + 4 overhead = 260 bytes = 2080 bit <= 4096 bit -> CUKUP
        assert sufficient is True
        assert max_cap == 4096
        assert req == 2080

    def test_save_and_reload_image(self, tmp_path):
        """Memverifikasi penyimpanan citra aman tanpa distorsi nilai piksel."""
        test_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        out_file = str(tmp_path / "test_out.png")
        
        save_image(test_img, out_file)
        assert os.path.exists(out_file)
        
        reloaded = load_cover_image(out_file)
        assert np.array_equal(test_img, reloaded)

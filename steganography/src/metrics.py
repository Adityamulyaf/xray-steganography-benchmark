"""
Modul Penghitungan Metrik Kualitas Citra dan Evaluasi Ekstraksi Steganografi.
Memenuhi Persyaratan FR-06, FR-08, FR-09, dan Bab 8 PRD.md.
"""

import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

from .dataset import bytes_to_bits


def calculate_mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """
    Menghitung Mean Squared Error (MSE) antara dua citra grayscale.
    
    MSE = (1 / MN) * sum((I - K)^2)
    
    Args:
        image_a: Citra referensi (uint8 / float).
        image_b: Citra hasil/pembanding (uint8 / float).
        
    Returns:
        float: Nilai MSE. Semakin kecil semakin identik kedua citra.
        
    Raises:
        ValueError: Jika dimensi citra tidak sama.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Dimensi citra tidak cocok untuk MSE: {image_a.shape} vs {image_b.shape}"
        )
        
    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    mse = np.mean(diff ** 2)
    return float(mse)


def calculate_psnr(image_a: np.ndarray, image_b: np.ndarray, max_val: float = 255.0) -> float:
    """
    Menghitung Peak Signal-to-Noise Ratio (PSNR) dalam decibel (dB).
    
    PSNR = 10 * log10(255^2 / MSE)
    Jika MSE == 0, mengembalikan float('inf').
    
    Args:
        image_a: Citra referensi.
        image_b: Citra hasil.
        max_val: Nilai puncak piksel (default 255.0 untuk 8-bit).
        
    Returns:
        float: Nilai PSNR dalam dB atau float('inf').
    """
    mse = calculate_mse(image_a, image_b)
    if mse == 0.0:
        return float("inf")
    psnr = 10.0 * math.log10((max_val ** 2) / mse)
    return float(psnr)


def calculate_ssim(image_a: np.ndarray, image_b: np.ndarray, data_range: float = 255.0) -> float:
    """
    Menghitung Structural Similarity Index (SSIM) antara dua citra.
    
    Args:
        image_a: Citra referensi (2D grayscale).
        image_b: Citra hasil (2D grayscale).
        data_range: Rentang nilai piksel (255.0 untuk 8-bit).
        
    Returns:
        float: Nilai SSIM dalam rentang [-1, 1], di mana 1.0 berarti identik struktural.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Dimensi citra tidak cocok untuk SSIM: {image_a.shape} vs {image_b.shape}"
        )
        
    score = ssim_fn(image_a, image_b, data_range=data_range)
    return float(score)


def calculate_bpp(payload_bytes_len: int, image_shape: Tuple[int, int]) -> float:
    """
    Menghitung kepadatan payload dalam Bits Per Pixel (BPP).
    
    BPP = (jumlah bit payload) / (W * H)
    Catatan: Overhead header tidak dihitung sebagai payload pengguna (PRD 8.6).
    
    Args:
        payload_bytes_len: Ukuran payload pengguna dalam bytes.
        image_shape: Tuple (Height, Width) citra.
        
    Returns:
        float: Nilai BPP.
    """
    h, w = image_shape
    total_pixels = h * w
    if total_pixels == 0:
        return 0.0
    payload_bits = payload_bytes_len * 8
    return float(payload_bits / total_pixels)


def calculate_ber(
    original_data: Union[bytes, np.ndarray],
    recovered_data: Optional[Union[bytes, np.ndarray]],
    expected_bits_len: Optional[int] = None
) -> float:
    """
    Menghitung Bit Error Rate (BER) antara pesan asli dan pesan hasil ekstraksi.
    
    BER = (jumlah bit salah) / (jumlah bit yang dibandingkan)
    
    Sesuai PRD 8.4:
    - Jika recovered_data adalah None atau kosong, BER = 1.0 (kegagalan total / 100% error).
    - Jika panjang bit berbeda, bit yang hilang atau berlebih dihitung sebagai bit error.
    
    Args:
        original_data: Bytes pesan asli atau bit array aslinya.
        recovered_data: Bytes pesan terekstrak atau None jika gagal ekstrak.
        expected_bits_len: Panjang bit yang diharapkan jika diketahui.
        
    Returns:
        float: Nilai rasio error bit dalam rentang [0.0, 1.0].
    """
    if recovered_data is None:
        return 1.0
        
    # Konversi ke bit array jika masih berupa bytes
    orig_bits = bytes_to_bits(original_data) if isinstance(original_data, bytes) else original_data
    rec_bits = bytes_to_bits(recovered_data) if isinstance(recovered_data, bytes) else recovered_data
    
    total_expected = len(orig_bits) if expected_bits_len is None else expected_bits_len
    if total_expected == 0:
        return 0.0 if len(rec_bits) == 0 else 1.0
        
    if len(rec_bits) == 0:
        return 1.0
        
    min_len = min(len(orig_bits), len(rec_bits))
    # Bit error pada bagian yang dapat disejajarkan
    bit_mismatches = np.count_nonzero(orig_bits[:min_len] != rec_bits[:min_len])
    # Selisih panjang dihitung sebagai error
    length_penalty = abs(len(orig_bits) - len(rec_bits))
    total_errors = bit_mismatches + length_penalty
    
    ber = min(1.0, total_errors / total_expected)
    return float(ber)


def calculate_image_metrics(image_ref: np.ndarray, image_test: np.ndarray) -> Dict[str, float]:
    """
    Menghitung seluruh metrik kualitas citra (MSE, PSNR, SSIM).
    
    Args:
        image_ref: Citra referensi (cover atau stego awal).
        image_test: Citra yang diuji (stego atau stego terkompresi).
        
    Returns:
        Dict[str, float]: Berisi kunci 'mse', 'psnr', dan 'ssim'.
    """
    mse = calculate_mse(image_ref, image_test)
    psnr = calculate_psnr(image_ref, image_test)
    ssim = calculate_ssim(image_ref, image_test)
    return {
        "mse": round(mse, 6),
        "psnr": round(psnr, 4) if not math.isinf(psnr) else float("inf"),
        "ssim": round(ssim, 6)
    }


def calculate_message_metrics(
    original_bytes: bytes,
    recovered_bytes: Optional[bytes]
) -> Dict[str, Any]:
    """
    Menghitung metrik ketahanan ekstraksi pesan (BER dan Extraction Success).
    
    Args:
        original_bytes: Data byte pesan referensi yang disisipkan.
        recovered_bytes: Data byte hasil ekstraksi atau None jika ekstraksi crash/gagal.
        
    Returns:
        Dict[str, Any]:
            - 'ber': Nilai float BER [0.0, 1.0].
            - 'extraction_success': Boolean (True jika identik 100%, False jika ada error).
    """
    if recovered_bytes is None:
        return {
            "ber": 1.0,
            "extraction_success": False
        }
        
    is_exact_match = (original_bytes == recovered_bytes)
    ber = 0.0 if is_exact_match else calculate_ber(original_bytes, recovered_bytes)
    
    return {
        "ber": round(ber, 6),
        "extraction_success": bool(is_exact_match)
    }

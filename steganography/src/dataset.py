"""
Modul Manajemen Dataset Citra X-Ray dan Pesan Rahasia Sintetis.
Memenuhi Persyaratan FR-01, FR-02, dan NFR-02 pada PRD.md.
"""

import os
from typing import Tuple, Union, Optional
import numpy as np
import cv2


def load_cover_image(image_path: str, force_grayscale: bool = True) -> np.ndarray:
    """
    Memuat dan memvalidasi citra cover X-Ray dari direktori lokal.
    
    Args:
        image_path: Jalur absolut atau relatif ke file citra.
        force_grayscale: Memastikan citra dibaca dalam mode 8-bit grayscale.
        
    Returns:
        np.ndarray: Citra grayscale 8-bit berdimensi (H, W) dengan dtype uint8.
        
    Raises:
        FileNotFoundError: Jika file citra tidak ditemukan.
        ValueError: Jika format citra tidak valid atau citra tidak dapat dibaca.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Citra cover tidak ditemukan pada: {image_path}")
    
    flag = cv2.IMREAD_GRAYSCALE if force_grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imread(image_path, flag)
    
    if image is None:
        raise ValueError(f"Gagal membaca citra atau file rusak: {image_path}")
    
    if len(image.shape) != 2:
        raise ValueError(
            f"Citra harus berupa 2D grayscale (ditemukan shape: {image.shape}). "
            "Gunakan citra grayscale 8-bit."
        )
        
    if image.dtype != np.uint8:
        raise ValueError(f"Tipe data citra harus uint8 (ditemukan: {image.dtype})")
        
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Menyimpan citra output (stego / compressed) ke direktori tujuan.
    Membuat direktori induk secara otomatis jika belum tersedia.
    
    Args:
        image: Array citra (np.uint8).
        output_path: Jalur penyimpanan file output.
        
    Raises:
        ValueError: Jika citra tidak valid.
        IOError: Jika gagal menyimpan file citra.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Citra yang akan disimpan tidak valid.")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Gagal menulis file citra ke: {output_path}")


def load_secret_message(message_path: str) -> bytes:
    """
    Membaca pesan rahasia pasien sintetis dalam bentuk bytes (UTF-8).
    
    Args:
        message_path: Jalur ke file pesan (.txt).
        
    Returns:
        bytes: Pesan rahasia dalam representasi bytes mentah.
        
    Raises:
        FileNotFoundError: Jika file pesan tidak ditemukan.
    """
    if not os.path.isfile(message_path):
        raise FileNotFoundError(f"File pesan tidak ditemukan: {message_path}")
        
    with open(message_path, "rb") as f:
        data = f.read()
    return data


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Mengonversi data bytes mentah menjadi array 1D bit (0 atau 1) dengan dtype uint8.
    Urutan bit: Most Significant Bit (MSB) first per byte.
    
    Args:
        data: Data bytes yang akan diubah.
        
    Returns:
        np.ndarray: Array 1D berisi angka 0 dan 1, panjang = len(data) * 8.
    """
    if len(data) == 0:
        return np.array([], dtype=np.uint8)
        
    byte_array = np.frombuffer(data, dtype=np.uint8)
    # unpackbits mengonversi tiap byte ke 8 bit (MSB first)
    bits = np.unpackbits(byte_array)
    return bits.astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Mengonversi array bit (0 atau 1) kembali ke representasi bytes mentah.
    
    Args:
        bits: Array 1D berisi bit 0 dan 1. Panjang harus kelipatan 8.
        
    Returns:
        bytes: Hasil konversi data biner ke bytes.
        
    Raises:
        ValueError: Jika panjang bits bukan kelipatan 8.
    """
    if len(bits) == 0:
        return b""
        
    if len(bits) % 8 != 0:
        raise ValueError(
            f"Panjang array bit ({len(bits)}) harus merupakan kelipatan 8 untuk dikonversi ke bytes."
        )
        
    # packbits mengemas 8 bit (MSB first) menjadi 1 byte
    byte_array = np.packbits(bits)
    return bytes(byte_array)


def check_capacity(
    image_shape: Tuple[int, int],
    payload_bytes_len: int,
    method: str = "lsb",
    overhead_bytes: int = 4
) -> Tuple[bool, int, int]:
    """
    Memeriksa apakah kapasitas citra mencukupi untuk menyisipkan payload beserta header.
    
    Args:
        image_shape: Tuple (Height, Width) citra.
        payload_bytes_len: Panjang pesan dalam bytes.
        method: Nama metode ('lsb', 'adaptive_lsb', 'dct').
        overhead_bytes: Jumlah byte overhead untuk header panjang pesan (default 4 byte / 32 bit).
        
    Returns:
        Tuple[bool, int, int]:
            - is_sufficient: True jika kapasitas mencukupi.
            - required_bits: Jumlah bit yang diperlukan (payload + overhead).
            - max_capacity_bits: Batas maksimum bit yang dapat ditampung.
    """
    h, w = image_shape
    total_pixels = h * w
    required_bits = (payload_bytes_len + overhead_bytes) * 8
    
    method_lower = method.lower()
    if method_lower in ["lsb", "adaptive_lsb"]:
        # 1 bit per piksel pada LSB konvensional
        max_capacity_bits = total_pixels
    elif method_lower == "dct":
        # Pada DCT blok 8x8, tiap blok dapat menyisipkan 1 bit
        blocks_count = (h // 8) * (w // 8)
        max_capacity_bits = blocks_count
    else:
        raise ValueError(f"Metode steganografi '{method}' tidak dikenali.")
        
    is_sufficient = required_bits <= max_capacity_bits
    return is_sufficient, required_bits, max_capacity_bits

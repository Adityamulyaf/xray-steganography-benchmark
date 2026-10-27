"""
Modul Implementasi Steganografi Least Significant Bit (LSB) Konvensional.
Memenuhi Persyaratan FR-03, FR-04, FR-10, dan Milestone 1 PRD.md.
"""

import os
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    from .dataset import (
        bytes_to_bits,
        bits_to_bytes,
        check_capacity,
        load_cover_image,
        save_image,
        load_secret_message
    )
    from .metrics import (
        calculate_image_metrics,
        calculate_message_metrics,
        calculate_bpp
    )
except ImportError:
    from dataset import (
        bytes_to_bits,
        bits_to_bytes,
        check_capacity,
        load_cover_image,
        save_image,
        load_secret_message
    )
    from metrics import (
        calculate_image_metrics,
        calculate_message_metrics,
        calculate_bpp
    )


HEADER_SIZE_BYTES = 4  # 32-bit unsigned integer untuk mencatat panjang payload dalam bytes
HEADER_BITS_COUNT = HEADER_SIZE_BYTES * 8


def embed_lsb(
    cover_image: np.ndarray,
    payload: bytes
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Menyisipkan payload ke dalam Least Significant Bit (LSB) citra grayscale secara sekuensial.
    
    Header protokol:
    - 32-bit pertama (4 byte, big-endian) menyimpan panjang payload (dalam bytes).
    - Bit berikutnya menyimpan bit payload asli.
    
    Args:
        cover_image: Citra cover 2D grayscale (uint8).
        payload: Data byte yang akan disisipkan.
        
    Returns:
        Tuple[np.ndarray, float, Dict[str, Any]]:
            - stego_image: Citra stego hasil penyisipan (np.uint8, copy independen).
            - embedding_time: Waktu penyisipan dalam detik.
            - info: Metadata proses embedding (panjang payload, status, kapasitas).
            
    Raises:
        ValueError: Jika payload melebihi kapasitas citra atau citra tidak valid.
    """
    start_time = time.perf_counter()
    
    if cover_image is None or len(cover_image.shape) != 2:
        raise ValueError("Cover image harus berupa citra 2D grayscale.")
        
    h, w = cover_image.shape
    payload_len = len(payload)
    
    # 1. Periksa batas kapasitas
    is_sufficient, required_bits, max_capacity = check_capacity(
        (h, w), payload_len, method="lsb", overhead_bytes=HEADER_SIZE_BYTES
    )
    if not is_sufficient:
        raise ValueError(
            f"Payload ({payload_len} byte) melebihi kapasitas citra. "
            f"Diperlukan {required_bits} bit, kapasitas maksimum {max_capacity} bit."
        )
        
    # 2. Siapkan header (4 bytes panjang payload) + data payload
    header_bytes = payload_len.to_bytes(HEADER_SIZE_BYTES, byteorder="big", signed=False)
    header_bits = bytes_to_bits(header_bytes)
    payload_bits = bytes_to_bits(payload)
    
    total_bits_to_embed = np.concatenate([header_bits, payload_bits])
    
    # 3. Lakukan modifikasi LSB secara terisolasi (tidak mengubah cover_image asli)
    stego_flat = cover_image.flatten().copy()
    num_bits = len(total_bits_to_embed)
    
    # Clear bit LSB (AND dengan 0xFE / 254) lalu set dengan bit pesan (OR)
    stego_flat[:num_bits] = (stego_flat[:num_bits] & 0xFE) | total_bits_to_embed
    stego_image = stego_flat.reshape((h, w))
    
    elapsed_time = time.perf_counter() - start_time
    
    info = {
        "method": "lsb",
        "payload_bytes": payload_len,
        "header_bits": HEADER_BITS_COUNT,
        "total_bits_embedded": num_bits,
        "max_capacity_bits": max_capacity,
        "embedding_time_seconds": elapsed_time,
        "success": True
    }
    
    return stego_image, elapsed_time, info


def extract_lsb(
    stego_image: np.ndarray
) -> Tuple[Optional[bytes], float, Dict[str, Any]]:
    """
    Mengekstrak payload dari Least Significant Bit (LSB) citra stego.
    Bersifat blind (tidak memerlukan citra cover referensi).
    
    Args:
        stego_image: Citra stego 2D grayscale (uint8).
        
    Returns:
        Tuple[Optional[bytes], float, Dict[str, Any]]:
            - recovered_payload: Bytes data hasil ekstraksi (atau None jika header rusak/invalid).
            - extraction_time: Waktu ekstraksi dalam detik.
            - info: Metadata proses ekstraksi (status, alasan jika gagal).
    """
    start_time = time.perf_counter()
    
    if stego_image is None or len(stego_image.shape) != 2:
        return None, 0.0, {
            "method": "lsb",
            "success": False,
            "error": "Citra stego tidak valid atau bukan 2D grayscale."
        }
        
    stego_flat = stego_image.flatten()
    total_pixels = len(stego_flat)
    
    # Periksa apakah citra memiliki cukup piksel untuk membaca header
    if total_pixels < HEADER_BITS_COUNT:
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, {
            "method": "lsb",
            "success": False,
            "error": f"Ukuran citra ({total_pixels} px) lebih kecil dari ukuran header ({HEADER_BITS_COUNT} bit)."
        }
        
    # 1. Ekstrak header (32 bit pertama)
    header_bits = stego_flat[:HEADER_BITS_COUNT] & 0x01
    try:
        header_bytes = bits_to_bytes(header_bits)
        payload_len = int.from_bytes(header_bytes, byteorder="big", signed=False)
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, {
            "method": "lsb",
            "success": False,
            "error": f"Gagal mendekode header: {str(e)}"
        }
        
    # 2. Validasi panjang payload yang terbaca dari header
    required_bits = HEADER_BITS_COUNT + (payload_len * 8)
    if payload_len == 0:
        elapsed_time = time.perf_counter() - start_time
        return b"", elapsed_time, {
            "method": "lsb",
            "payload_bytes": 0,
            "extraction_time_seconds": elapsed_time,
            "success": True
        }
        
    if required_bits > total_pixels or payload_len > (total_pixels // 8):
        # Header kemungkinan besar rusak akibat kompresi atau stego tidak valid
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, {
            "method": "lsb",
            "success": False,
            "error": (
                f"Header mengindikasikan payload {payload_len} bytes ({required_bits} bit), "
                f"melebihi ukuran citra {total_pixels} piksel."
            )
        }
        
    # 3. Ekstrak bit payload
    payload_bits = stego_flat[HEADER_BITS_COUNT:required_bits] & 0x01
    recovered_payload = bits_to_bytes(payload_bits)
    
    elapsed_time = time.perf_counter() - start_time
    
    info = {
        "method": "lsb",
        "payload_bytes": len(recovered_payload),
        "total_bits_extracted": len(payload_bits),
        "extraction_time_seconds": elapsed_time,
        "success": True
    }
    
    return recovered_payload, elapsed_time, info


if __name__ == "__main__":
    print("=" * 65)
    print("DEMO EKSPERIMEN LSB KONVENSIONAL (UJI COBA OUTPUT NYATA)")
    print("=" * 65)

    cover_file = "data/cover/xray_sample_01.png"
    secret_file = "data/secret/sample_256bytes.txt"
    stego_out_file = "outputs/stego/stego_lsb_xray01_256bytes.png"
    recovered_out_file = "outputs/recovered/recovered_lsb_xray01_256bytes.txt"

    print(f"\n[1] Memuat Citra Cover: {cover_file}")
    cover = load_cover_image(cover_file)
    print(f"    - Resolusi: {cover.shape[1]}x{cover.shape[0]} piksel")
    print(f"    - Format: Grayscale 8-bit ({cover.dtype})")

    print(f"\n[2] Memuat Pesan Rahasia Sintetis: {secret_file}")
    secret = load_secret_message(secret_file)
    print(f"    - Panjang Pesan: {len(secret)} bytes ({len(secret) * 8} bit)")
    print(f"    - Cuplikan Pesan: {secret[:50].decode('utf-8', errors='ignore')}...")

    print("\n[3] Menjalankan Penyisipan (Embedding)...")
    stego_img, embed_t, embed_info = embed_lsb(cover, secret)
    save_image(stego_img, stego_out_file)
    print(f"    - Stego Image Berhasil Disimpan ke: {stego_out_file}")
    print(f"    - Waktu Embedding: {embed_t * 1000:.3f} ms")

    print(f"\n[4] Memuat Citra Stego dari Disk & Menjalankan Ekstraksi...")
    loaded_stego = load_cover_image(stego_out_file)
    recovered, extract_t, extract_info = extract_lsb(loaded_stego)
    
    with open(recovered_out_file, "wb") as f:
        f.write(recovered if recovered is not None else b"")
    print(f"    - Pesan Terekstrak Disimpan ke: {recovered_out_file}")
    print(f"    - Waktu Ekstraksi: {extract_t * 1000:.3f} ms")

    print("\n[5] Evaluasi Metrik Kualitas & Keberhasilan:")
    img_met = calculate_image_metrics(cover, loaded_stego)
    msg_met = calculate_message_metrics(secret, recovered)
    bpp = calculate_bpp(len(secret), cover.shape)

    print("-" * 65)
    print(f"  MSE (Mean Squared Error)          : {img_met['mse']:.6f}")
    print(f"  PSNR (Peak Signal-to-Noise Ratio) : {img_met['psnr']:.2f} dB")
    print(f"  SSIM (Structural Similarity)      : {img_met['ssim']:.6f}")
    print(f"  BPP (Bits Per Pixel)              : {bpp:.6f} bpp")
    print(f"  BER (Bit Error Rate)              : {msg_met['ber']:.6f}")
    print(f"  Extraction Success (ESR)          : {'BERHASIL 100%' if msg_met['extraction_success'] else 'GAGAL'}")
    print("-" * 65)
    print("Demo selesai. File output dapat diperiksa di folder outputs/stego dan outputs/recovered.\n")


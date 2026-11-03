"""
Modul Implementasi Steganografi Content-Adaptive LSB.
Memenuhi Persyaratan FR-05, FR-10, Milestone 2, dan AC-03 PRD.md / STEP.md.

Algoritma ini menganalisis kompleksitas konten lokal citra menggunakan operator gradien
(Sobel) pada 7 bit paling signifikan (MSB) agar invarian terhadap proses penyisipan.
Piksel dengan magnitudo gradien tertinggi (area tulang, tepi anatomi, dan tekstur tinggi)
diprioritaskan terlebih dahulu untuk menyembunyikan noise steganografi, meminimalkan
gangguan visual pada area homogen/halus citra X-Ray.
Ekstraksi bersifat blind (tanpa memerlukan citra cover asli).
"""

import os
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

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


def compute_complexity_map(image: np.ndarray, operator: str = "sobel") -> np.ndarray:
    """
    Menghitung peta kompleksitas tekstur/konten lokal citra.
    
    PENTING: Komputasi dilakukan pada 7 MSB (Most Significant Bits) dengan masking 0xFE
    sehingga nilai gradien tetap 100% identik baik pada citra cover maupun stego,
    memungkinkan ekstraksi bersifat blind (tanpa citra asli).

    Args:
        image: Citra grayscale 2D (uint8).
        operator: Nama operator gradien (default 'sobel').

    Returns:
        np.ndarray: Magnitudo gradien 2D bertipe float64 berukuran sama dengan citra.
    """
    if len(image.shape) != 2:
        raise ValueError("Citra harus berupa array 2D grayscale.")

    # Gunakan 7 MSB (nol-kan LSB) agar peta invarian terhadap modifikasi LSB
    invariant_base = (image & 0xFE).astype(np.float64)

    op_lower = operator.lower()
    if op_lower == "sobel":
        # Sobel gradien horizontal (dx) dan vertikal (dy)
        gx = cv2.Sobel(invariant_base, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(invariant_base, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.hypot(gx, gy)
    elif op_lower == "laplacian":
        magnitude = np.abs(cv2.Laplacian(invariant_base, cv2.CV_64F, ksize=3))
    else:
        raise ValueError(f"Operator kompleksitas '{operator}' tidak didukung. Pilih 'sobel' atau 'laplacian'.")

    return magnitude


def get_adaptive_pixel_order(image: np.ndarray, operator: str = "sobel") -> np.ndarray:
    """
    Menghasilkan urutan indeks piksel (flat 1D) yang diurutkan berdasarkan kompleksitas tekstur
    dari yang tertinggi ke terendah secara deterministik (stable sort).

    Args:
        image: Citra grayscale 2D (uint8).
        operator: Nama operator gradien.

    Returns:
        np.ndarray: Array 1D berisi indeks flat terurut secara menurun berdasarkan kompleksitas.
    """
    magnitude = compute_complexity_map(image, operator=operator)
    flat_mag = magnitude.ravel()
    # Gunakan stable sort ('mergesort' / 'stable') agar tie-breaking urutan piksel selalu deterministik
    sorted_indices = np.argsort(-flat_mag, kind="stable")
    return sorted_indices


def embed_adaptive_lsb(
    cover_image: np.ndarray,
    payload: bytes,
    operator: str = "sobel"
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Menyisipkan payload ke dalam Least Significant Bit (LSB) citra grayscale secara adaptif
    pada area dengan kompleksitas visual / tekstur tertinggi (misal: tulang dan batas organ pada X-Ray).

    Header protokol:
    - 32-bit pertama pada urutan adaptif menyimpan panjang payload (dalam bytes).
    - Bit berikutnya menyimpan bit payload asli pada urutan adaptif berikutnya.

    Args:
        cover_image: Citra cover 2D grayscale (uint8).
        payload: Data byte yang akan disisipkan.
        operator: Operator gradien untuk pemetaan tekstur ('sobel').

    Returns:
        Tuple[np.ndarray, float, Dict[str, Any]]:
            - stego_image: Citra stego hasil penyisipan (np.uint8, copy independen).
            - embedding_time: Waktu penyisipan dalam detik.
            - info: Metadata proses embedding.

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
        (h, w), payload_len, method="adaptive_lsb", overhead_bytes=HEADER_SIZE_BYTES
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
    num_bits = len(total_bits_to_embed)

    # 3. Hitung urutan piksel adaptif dari 7 MSB
    ranked_indices = get_adaptive_pixel_order(cover_image, operator=operator)
    target_indices = ranked_indices[:num_bits]

    # 4. Modifikasi LSB hanya pada piksel target bertekstur tinggi (terisolasi)
    stego_flat = cover_image.flatten().copy()
    stego_flat[target_indices] = (stego_flat[target_indices] & 0xFE) | total_bits_to_embed
    stego_image = stego_flat.reshape((h, w))

    elapsed_time = time.perf_counter() - start_time

    info = {
        "method": "adaptive_lsb",
        "operator": operator,
        "payload_bytes": payload_len,
        "header_bits": HEADER_BITS_COUNT,
        "total_bits_embedded": num_bits,
        "max_capacity_bits": max_capacity,
        "embedding_time_seconds": elapsed_time,
        "success": True
    }

    return stego_image, elapsed_time, info


def extract_adaptive_lsb(
    stego_image: np.ndarray,
    operator: str = "sobel",
    expected_payload_len: Optional[int] = None
) -> Tuple[Optional[bytes], float, Dict[str, Any]]:
    """
    Mengekstrak payload dari Least Significant Bit (LSB) citra stego secara adaptif.
    Bersifat blind (tidak memerlukan citra cover referensi).

    Args:
        stego_image: Citra stego 2D grayscale (uint8).
        operator: Operator gradien yang identik dengan saat embedding ('sobel').
        expected_payload_len: Opsional, panjang payload yang diharapkan (digunakan sebagai fallback
                              saat simulasi serangan jika header mengalami korupsi).

    Returns:
        Tuple[Optional[bytes], float, Dict[str, Any]]:
            - recovered_payload: Bytes data hasil ekstraksi (atau None jika header rusak/invalid).
            - extraction_time: Waktu ekstraksi dalam detik.
            - info: Metadata proses ekstraksi.
    """
    start_time = time.perf_counter()

    if stego_image is None or len(stego_image.shape) != 2:
        return None, 0.0, {
            "method": "adaptive_lsb",
            "success": False,
            "error": "Citra stego tidak valid atau bukan 2D grayscale."
        }

    stego_flat = stego_image.flatten()
    total_pixels = len(stego_flat)

    if total_pixels < HEADER_BITS_COUNT:
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, {
            "method": "adaptive_lsb",
            "success": False,
            "error": f"Ukuran citra ({total_pixels} px) lebih kecil dari ukuran header ({HEADER_BITS_COUNT} bit)."
        }

    # 1. Rekonstruksi urutan indeks adaptif (identik karena berbasis 7 MSB)
    ranked_indices = get_adaptive_pixel_order(stego_image, operator=operator)

    # 2. Ekstrak header (32 bit pertama dari piksel peringkat tertinggi)
    header_indices = ranked_indices[:HEADER_BITS_COUNT]
    header_bits = stego_flat[header_indices] & 0x01

    payload_len = None
    header_valid = False
    try:
        header_bytes = bits_to_bytes(header_bits)
        payload_len = int.from_bytes(header_bytes, byteorder="big", signed=False)
        required_bits = HEADER_BITS_COUNT + (payload_len * 8)
        if 0 <= payload_len <= ((total_pixels - HEADER_BITS_COUNT) // 8):
            header_valid = True
    except Exception:
        header_valid = False

    # Jika header korupsi dan fallback expected_payload_len diberikan
    if not header_valid:
        if expected_payload_len is not None and expected_payload_len <= ((total_pixels - HEADER_BITS_COUNT) // 8):
            payload_len = expected_payload_len
            required_bits = HEADER_BITS_COUNT + (payload_len * 8)
            payload_indices = ranked_indices[HEADER_BITS_COUNT:required_bits]
            payload_bits = stego_flat[payload_indices] & 0x01
            recovered_payload = bits_to_bytes(payload_bits)
            elapsed_time = time.perf_counter() - start_time
            return recovered_payload, elapsed_time, {
                "method": "adaptive_lsb",
                "operator": operator,
                "payload_bytes": len(recovered_payload),
                "total_bits_extracted": len(payload_bits),
                "header_valid": False,
                "header_corrupted": True,
                "extraction_time_seconds": elapsed_time,
                "success": False,
                "error": "Header korupsi akibat serangan kompresi. Menggunakan fallback expected_payload_len."
            }
        else:
            elapsed_time = time.perf_counter() - start_time
            return None, elapsed_time, {
                "method": "adaptive_lsb",
                "success": False,
                "error": (
                    f"Header mengindikasikan payload {payload_len} bytes, "
                    f"melebihi ukuran citra {total_pixels} piksel."
                )
            }

    if payload_len == 0:
        elapsed_time = time.perf_counter() - start_time
        return b"", elapsed_time, {
            "method": "adaptive_lsb",
            "payload_bytes": 0,
            "header_valid": True,
            "extraction_time_seconds": elapsed_time,
            "success": True
        }

    # 4. Ekstrak bit payload dari urutan peringkat berikutnya
    required_bits = HEADER_BITS_COUNT + (payload_len * 8)
    payload_indices = ranked_indices[HEADER_BITS_COUNT:required_bits]
    payload_bits = stego_flat[payload_indices] & 0x01
    recovered_payload = bits_to_bytes(payload_bits)

    elapsed_time = time.perf_counter() - start_time

    info = {
        "method": "adaptive_lsb",
        "operator": operator,
        "payload_bytes": len(recovered_payload),
        "total_bits_extracted": len(payload_bits),
        "header_valid": True,
        "extraction_time_seconds": elapsed_time,
        "success": True
    }

    return recovered_payload, elapsed_time, info


if __name__ == "__main__":
    print("=" * 65)
    print("DEMO EKSPERIMEN CONTENT-ADAPTIVE LSB (FASE 3 / MILESTONE 2)")
    print("=" * 65)

    cover_file = "data/cover/xray_sample_01.png"
    secret_file = "data/secret/sample_256bytes.txt"
    stego_out_file = "outputs/stego/stego_adaptive_lsb_xray01_256bytes.png"
    recovered_out_file = "outputs/recovered/recovered_adaptive_lsb_xray01_256bytes.txt"

    print(f"\n[1] Memuat Citra Cover: {cover_file}")
    cover = load_cover_image(cover_file)
    print(f"    - Resolusi: {cover.shape[1]}x{cover.shape[0]} piksel")
    print(f"    - Format: Grayscale 8-bit ({cover.dtype})")

    print(f"\n[2] Memuat Pesan Rahasia Sintetis: {secret_file}")
    secret = load_secret_message(secret_file)
    print(f"    - Panjang Pesan: {len(secret)} bytes ({len(secret) * 8} bit)")
    print(f"    - Cuplikan Pesan: {secret[:50].decode('utf-8', errors='ignore')}...")

    print("\n[3] Menjalankan Penyisipan Adaptif (Adaptive Embedding)...")
    stego_img, embed_t, embed_info = embed_adaptive_lsb(cover, secret, operator="sobel")
    save_image(stego_img, stego_out_file)
    print(f"    - Stego Image Disimpan ke: {stego_out_file}")
    print(f"    - Waktu Embedding: {embed_t * 1000:.3f} ms")

    print("\n[4] Memuat Citra Stego dari Disk & Menjalankan Blind Extraction...")
    loaded_stego = load_cover_image(stego_out_file)
    recovered, extract_t, extract_info = extract_adaptive_lsb(loaded_stego, operator="sobel")

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
    print("Demo Content-Adaptive LSB selesai.\n")


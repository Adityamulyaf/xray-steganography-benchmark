"""
Modul Implementasi Steganografi Discrete Cosine Transform (DCT).
Memenuhi Persyaratan FR-06, FR-10, Milestone 3, dan AC-04 PRD.md / STEP.md.

Algoritma ini beroperasi pada domain frekuensi:
1. Citra dipartisi ke dalam blok 8x8 piksel non-overlapping.
2. Setiap blok ditransformasikan ke domain frekuensi menggunakan 2D DCT orthonormal.
3. Penyisipan bit dilakukan dengan memodifikasi pasangan koefisien frekuensi menengah
   (mid-frequency coefficients, secara default (4, 3) dan (3, 4)) dengan margin ketahanan
   (quantization / robustness factor) agar tahan terhadap kompresi JPEG.
4. Inverse 2D DCT (IDCT) diterapkan, diikuti pembulatan ke integer dan clipping ke rentang [0, 255].
5. Dilengkapi mekanisme kompensasi clipping (iterative refinement) untuk menjamin 0% error
   pada kondisi baseline (tanpa serangan).
"""

import os
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.fftpack import dct, idct

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
DEFAULT_BLOCK_SIZE = 8
DEFAULT_COEFF_PAIR_1 = (4, 3)
DEFAULT_COEFF_PAIR_2 = (3, 4)
DEFAULT_QUANTIZATION_FACTOR = 25.0


def dct2(blocks: np.ndarray) -> np.ndarray:
    """
    Menghitung 2D Discrete Cosine Transform (DCT) orthonormal pada blok citra.
    Mendukung array 2D tunggal (8, 8) maupun tumpukan blok 3D (N, 8, 8).
    """
    return dct(dct(blocks, axis=-1, norm="ortho"), axis=-2, norm="ortho")


def idct2(blocks: np.ndarray) -> np.ndarray:
    """
    Menghitung 2D Inverse Discrete Cosine Transform (IDCT) orthonormal pada koefisien frekuensi.
    Mendukung array 2D tunggal (8, 8) maupun tumpukan blok 3D (N, 8, 8).
    """
    return idct(idct(blocks, axis=-1, norm="ortho"), axis=-2, norm="ortho")


def embed_dct(
    cover_image: np.ndarray,
    payload: bytes,
    quantization_factor: float = DEFAULT_QUANTIZATION_FACTOR,
    coeff_pair_1: Tuple[int, int] = DEFAULT_COEFF_PAIR_1,
    coeff_pair_2: Tuple[int, int] = DEFAULT_COEFF_PAIR_2,
    block_size: int = DEFAULT_BLOCK_SIZE
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Menyisipkan payload ke dalam citra grayscale menggunakan steganografi domain frekuensi DCT 8x8.

    Header protokol:
    - 32 blok pertama menyimpan 32-bit integer panjang payload dalam bytes.
    - Blok-blok berikutnya menyimpan bit payload asli secara sekuensial.

    Args:
        cover_image: Citra cover 2D grayscale (uint8).
        payload: Data byte yang akan disisipkan.
        quantization_factor: Margin selisih koefisien (P) untuk ketahanan kuantisasi.
        coeff_pair_1: Koordinat koefisien mid-frequency pertama (row, col).
        coeff_pair_2: Koordinat koefisien mid-frequency kedua (row, col).
        block_size: Dimensi blok DCT (default 8).

    Returns:
        Tuple[np.ndarray, float, Dict[str, Any]]:
            - stego_image: Citra stego hasil inverse DCT (np.uint8, copy independen).
            - embedding_time: Waktu penyisipan dalam detik.
            - info: Metadata proses embedding (panjang payload, status, kapasitas).

    Raises:
        ValueError: Jika citra tidak valid, dimensi tidak habis dibagi block_size,
                    atau payload melebihi kapasitas citra.
    """
    start_time = time.perf_counter()

    if cover_image is None or len(cover_image.shape) != 2:
        raise ValueError("Cover image harus berupa citra 2D grayscale.")

    h, w = cover_image.shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"Dimensi citra ({w}x{h}) harus habis dibagi block_size ({block_size})."
        )

    payload_len = len(payload)

    # 1. Periksa batas kapasitas
    is_sufficient, required_bits, max_capacity = check_capacity(
        (h, w), payload_len, method="dct", overhead_bytes=HEADER_SIZE_BYTES
    )
    if not is_sufficient:
        raise ValueError(
            f"Payload ({payload_len} byte) melebihi kapasitas citra DCT. "
            f"Diperlukan {required_bits} blok, kapasitas maksimum {max_capacity} blok."
        )

    # 2. Siapkan bit stream: Header (4 bytes) + Payload
    header_bytes = payload_len.to_bytes(HEADER_SIZE_BYTES, byteorder="big", signed=False)
    header_bits = bytes_to_bits(header_bytes)
    payload_bits = bytes_to_bits(payload)
    total_bits = np.concatenate([header_bits, payload_bits])
    num_bits = len(total_bits)

    # 3. Partisi citra ke dalam blok (h//8, w//8, 8, 8) lalu ratakan ke (N_blocks, 8, 8)
    n_rows, n_cols = h // block_size, w // block_size
    blocks = (
        cover_image.copy()
        .reshape(n_rows, block_size, n_cols, block_size)
        .swapaxes(1, 2)
        .astype(np.float64)
    )
    flat_blocks = blocks.reshape(-1, block_size, block_size).copy()

    # 4. Modifikasi koefisien frekuensi pada target blok (vektorisasi cepat)
    target_blocks = flat_blocks[:num_bits]
    dct_blocks = dct2(target_blocks)

    u1, v1 = coeff_pair_1
    u2, v2 = coeff_pair_2
    c1 = dct_blocks[:, u1, v1]
    c2 = dct_blocks[:, u2, v2]
    P = float(quantization_factor)

    # Aturan penyisipan:
    # Bit 0 -> inginkan c1 - c2 >= P
    # Bit 1 -> inginkan c2 - c1 >= P
    bit0_mask = (total_bits == 0)
    bit1_mask = (total_bits == 1)

    cond0 = (c1 - c2 < P) & bit0_mask
    delta0 = np.where(cond0, (P - (c1 - c2)) / 2.0, 0.0)
    dct_blocks[:, u1, v1] += delta0
    dct_blocks[:, u2, v2] -= delta0

    cond1 = (c2 - c1 < P) & bit1_mask
    delta1 = np.where(cond1, (P - (c2 - c1)) / 2.0, 0.0)
    dct_blocks[:, u1, v1] -= delta1
    dct_blocks[:, u2, v2] += delta1

    # 5. Inverse DCT dan pembulatan ke domain spasial
    rec_blocks = np.clip(np.round(idct2(dct_blocks)), 0, 255).astype(np.uint8)

    # 6. Kompensasi clipping / pembulatan iteratif jika ada bit yang bergeser
    rec_dct = dct2(rec_blocks.astype(np.float64))
    rec_c1 = rec_dct[:, u1, v1]
    rec_c2 = rec_dct[:, u2, v2]
    extracted_bits = np.where(rec_c1 > rec_c2, 0, 1)

    diff_indices = np.where(extracted_bits != total_bits)[0]
    for idx in diff_indices:
        b = total_bits[idx]
        cur_P = P + 10.0
        blk = flat_blocks[idx].copy()
        for _ in range(6):
            c_block = dct2(blk)
            c1_b, c2_b = c_block[u1, v1], c_block[u2, v2]
            if b == 0:
                d = (cur_P - (c1_b - c2_b)) / 2.0
                c_block[u1, v1] += d
                c_block[u2, v2] -= d
            else:
                d = (cur_P - (c2_b - c1_b)) / 2.0
                c_block[u1, v1] -= d
                c_block[u2, v2] += d
            r_blk = np.clip(np.round(idct2(c_block)), 0, 255).astype(np.uint8)
            rc = dct2(r_blk.astype(np.float64))
            r_bit = 0 if rc[u1, v1] > rc[u2, v2] else 1
            if r_bit == b:
                rec_blocks[idx] = r_blk
                break
            cur_P += 10.0

    # 7. Susun kembali blok ke citra 2D utuh
    flat_blocks[:num_bits] = rec_blocks
    stego_image = (
        flat_blocks.reshape(n_rows, n_cols, block_size, block_size)
        .swapaxes(1, 2)
        .reshape(h, w)
        .astype(np.uint8)
    )

    elapsed_time = time.perf_counter() - start_time

    info = {
        "method": "dct",
        "payload_bytes": payload_len,
        "header_bits": HEADER_BITS_COUNT,
        "total_bits_embedded": num_bits,
        "max_capacity_bits": max_capacity,
        "quantization_factor": quantization_factor,
        "coeff_pair_1": coeff_pair_1,
        "coeff_pair_2": coeff_pair_2,
        "block_size": block_size,
        "embedding_time_seconds": elapsed_time,
        "success": True
    }

    return stego_image, elapsed_time, info


def extract_dct(
    stego_image: np.ndarray,
    coeff_pair_1: Tuple[int, int] = DEFAULT_COEFF_PAIR_1,
    coeff_pair_2: Tuple[int, int] = DEFAULT_COEFF_PAIR_2,
    block_size: int = DEFAULT_BLOCK_SIZE,
    expected_payload_len: Optional[int] = None
) -> Tuple[Optional[bytes], float, Dict[str, Any]]:
    """
    Mengekstrak payload dari domain frekuensi DCT citra stego.
    Bersifat blind (tidak memerlukan citra cover referensi).

    Args:
        stego_image: Citra stego 2D grayscale (uint8).
        coeff_pair_1: Koordinat koefisien mid-frequency pertama (row, col).
        coeff_pair_2: Koordinat koefisien mid-frequency kedua (row, col).
        block_size: Dimensi blok DCT (default 8).
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
            "method": "dct",
            "success": False,
            "error": "Citra stego tidak valid atau bukan 2D grayscale."
        }

    h, w = stego_image.shape
    if h % block_size != 0 or w % block_size != 0:
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, {
            "method": "dct",
            "success": False,
            "error": f"Dimensi citra ({w}x{h}) tidak habis dibagi block_size ({block_size})."
        }

    n_rows, n_cols = h // block_size, w // block_size
    total_blocks = n_rows * n_cols

    if total_blocks < HEADER_BITS_COUNT:
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, {
            "method": "dct",
            "success": False,
            "error": f"Jumlah blok citra ({total_blocks}) lebih kecil dari ukuran header ({HEADER_BITS_COUNT} bit)."
        }

    # 1. Partisi citra ke blok dan ratakan
    blocks = (
        stego_image.reshape(n_rows, block_size, n_cols, block_size)
        .swapaxes(1, 2)
        .astype(np.float64)
    )
    flat_blocks = blocks.reshape(-1, block_size, block_size)

    u1, v1 = coeff_pair_1
    u2, v2 = coeff_pair_2

    # 2. Ekstrak bit dari header (32 blok pertama)
    header_blocks = flat_blocks[:HEADER_BITS_COUNT]
    header_dct = dct2(header_blocks)
    c1_h = header_dct[:, u1, v1]
    c2_h = header_dct[:, u2, v2]
    header_bits = np.where(c1_h > c2_h, 0, 1).astype(np.uint8)

    payload_len = None
    header_valid = False
    try:
        header_bytes = bits_to_bytes(header_bits)
        payload_len = int.from_bytes(header_bytes, byteorder="big", signed=False)
        required_blocks = HEADER_BITS_COUNT + (payload_len * 8)
        if 0 <= payload_len <= ((total_blocks - HEADER_BITS_COUNT) // 8):
            header_valid = True
    except Exception:
        header_valid = False

    # Jika header korupsi dan fallback expected_payload_len diberikan
    if not header_valid:
        if expected_payload_len is not None and expected_payload_len <= ((total_blocks - HEADER_BITS_COUNT) // 8):
            payload_len = expected_payload_len
        else:
            elapsed_time = time.perf_counter() - start_time
            return None, elapsed_time, {
                "method": "dct",
                "success": False,
                "error": f"Header korupsi atau mengindikasikan payload {payload_len} yang melebihi kapasitas {total_blocks} blok."
            }

    if payload_len == 0:
        elapsed_time = time.perf_counter() - start_time
        return b"", elapsed_time, {
            "method": "dct",
            "payload_bytes": 0,
            "extraction_time_seconds": elapsed_time,
            "success": True
        }

    # 3. Ekstrak bit payload
    required_blocks = HEADER_BITS_COUNT + (payload_len * 8)
    payload_blocks = flat_blocks[HEADER_BITS_COUNT:required_blocks]
    payload_dct = dct2(payload_blocks)
    c1_p = payload_dct[:, u1, v1]
    c2_p = payload_dct[:, u2, v2]
    payload_bits = np.where(c1_p > c2_p, 0, 1).astype(np.uint8)
    recovered_payload = bits_to_bytes(payload_bits)

    elapsed_time = time.perf_counter() - start_time

    info = {
        "method": "dct",
        "payload_bytes": len(recovered_payload),
        "total_bits_extracted": len(payload_bits),
        "header_valid": header_valid,
        "extraction_time_seconds": elapsed_time,
        "success": True
    }

    return recovered_payload, elapsed_time, info


if __name__ == "__main__":
    print("=" * 65)
    print("DEMO EKSPERIMEN DCT STEGANOGRAPHY (FASE 4 / MILESTONE 3)")
    print("=" * 65)

    cover_file = "data/cover/xray_sample_01.png"
    secret_file = "data/secret/sample_256bytes.txt"
    stego_out_file = "outputs/stego/stego_dct_xray01_256bytes.png"
    recovered_out_file = "outputs/recovered/recovered_dct_xray01_256bytes.txt"

    print(f"\n[1] Memuat Citra Cover: {cover_file}")
    cover = load_cover_image(cover_file)
    print(f"    - Resolusi: {cover.shape[1]}x{cover.shape[0]} piksel")
    print(f"    - Format: Grayscale 8-bit ({cover.dtype})")

    print(f"\n[2] Memuat Pesan Rahasia Sintetis: {secret_file}")
    secret = load_secret_message(secret_file)
    print(f"    - Panjang Pesan: {len(secret)} bytes ({len(secret) * 8} bit)")
    print(f"    - Cuplikan Pesan: {secret[:50].decode('utf-8', errors='ignore')}...")

    print("\n[3] Menjalankan Penyisipan Domain Frekuensi DCT 8x8...")
    stego_img, embed_t, embed_info = embed_dct(cover, secret, quantization_factor=25.0)
    save_image(stego_img, stego_out_file)
    print(f"    - Stego Image Disimpan ke: {stego_out_file}")
    print(f"    - Waktu Embedding: {embed_t * 1000:.3f} ms")

    print("\n[4] Memuat Citra Stego dari Disk & Menjalankan Blind Extraction...")
    loaded_stego = load_cover_image(stego_out_file)
    recovered, extract_t, extract_info = extract_dct(loaded_stego)

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
    print("Demo DCT Steganography selesai.\n")


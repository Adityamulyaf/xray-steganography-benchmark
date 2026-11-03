"""
Modul Simulasi Serangan Kompresi JPEG (Milestone 4).
Memenuhi Persyaratan FR-05, FR-07, FR-09, FR-15, dan AC-05 PRD.md / STEP.md.

Modul ini mensimulasikan kompresi lossy JPEG pada berbagai faktor kualitas Q in {95, 90, 85, 80, 75},
menyimpan citra terkompresi ke direktori outputs/compressed/ secara terisolasi tanpa mengubah citra asli,
serta mengukur ketahanan pesan (BER, Extraction Success Rate) dan degradasi citra (PSNR, SSIM, MSE).
"""

import os
import time
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2

try:
    from .dataset import (
        load_cover_image,
        save_image,
        load_secret_message,
        check_capacity
    )
    from .metrics import (
        calculate_image_metrics,
        calculate_message_metrics,
        calculate_bpp
    )
    from .lsb import embed_lsb, extract_lsb
    from .adaptive_lsb import embed_adaptive_lsb, extract_adaptive_lsb
    from .dct import embed_dct, extract_dct
except ImportError:
    from dataset import (
        load_cover_image,
        save_image,
        load_secret_message,
        check_capacity
    )
    from metrics import (
        calculate_image_metrics,
        calculate_message_metrics,
        calculate_bpp
    )
    from lsb import embed_lsb, extract_lsb
    from adaptive_lsb import embed_adaptive_lsb, extract_adaptive_lsb
    from dct import embed_dct, extract_dct


DEFAULT_JPEG_QUALITIES: List[int] = [95, 90, 85, 80, 75]


def compress_jpeg(
    image: np.ndarray,
    quality: int,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Melakukan kompresi lossy JPEG pada citra grayscale 2D dengan faktor kualitas tertentu.
    
    Args:
        image: Citra 2D grayscale (uint8).
        quality: Faktor kualitas kompresi JPEG dalam rentang [1, 100].
        output_path: Opsional, jalur file untuk menyimpan citra JPEG hasil kompresi.

    Returns:
        Tuple[np.ndarray, int]:
            - compressed_image: Citra hasil kompresi dan dekompresi kembali (np.ndarray uint8 2D).
            - file_size_bytes: Ukuran data stream JPEG dalam bytes.

    Raises:
        ValueError: Jika faktor kualitas di luar rentang [1, 100] atau citra tidak valid.
    """
    if not (1 <= quality <= 100):
        raise ValueError(f"Faktor kualitas JPEG harus berada pada rentang 1-100, diterima: {quality}")

    if image is None or len(image.shape) != 2:
        raise ValueError("Citra harus berupa array 2D grayscale uint8.")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, encoded_buf = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError(f"Gagal melakukan encoding JPEG untuk kualitas {quality}.")

    file_size_bytes = len(encoded_buf)

    # Simpan ke disk jika output_path diberikan
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(encoded_buf.tobytes())

    # Decode kembali ke numpy array grayscale
    decompressed = cv2.imdecode(encoded_buf, cv2.IMREAD_GRAYSCALE)
    if decompressed is None:
        raise RuntimeError("Gagal mendekode kembali buffer JPEG.")

    return decompressed, file_size_bytes


def evaluate_jpeg_attack(
    cover_image: np.ndarray,
    stego_image: np.ndarray,
    method: str,
    original_payload: bytes,
    quality: int,
    compressed_output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Menjalankan simulasi serangan kompresi JPEG terhadap citra stego dan mengevaluasi seluruh
    metrik kualitas citra serta ketahanan ekstraksi pesan secara komprehensif.

    Menghitung 3 pasangan perbandingan kualitas citra:
    1. Cover vs Stego (distorsi embedding awal)
    2. Stego vs Stego Terkompresi (distorsi akibat serangan)
    3. Cover vs Stego Terkompresi (distorsi total)

    Args:
        cover_image: Citra referensi asli (uint8 2D).
        stego_image: Citra stego sebelum kompresi (uint8 2D).
        method: Nama metode steganografi ('lsb', 'adaptive_lsb', 'dct').
        original_payload: Bytes pesan asli yang disisipkan.
        quality: Faktor kualitas JPEG yang diuji.
        compressed_output_path: Jalur penyimpanan file hasil kompresi JPEG.
        **kwargs: Parameter tambahan untuk metode (misal: quantization_factor, operator).

    Returns:
        Dict[str, Any]: Metadata lengkap metrik citra dan hasil ekstraksi pesan.
    """
    method_lower = method.lower()
    payload_len = len(original_payload)

    # 1. Simulasikan kompresi JPEG
    compressed_stego, file_size = compress_jpeg(
        stego_image, quality=quality, output_path=compressed_output_path
    )

    # 2. Metrik Kualitas Citra (3 Pasang Perbandingan)
    metrics_cover_stego = calculate_image_metrics(cover_image, stego_image)
    metrics_stego_compressed = calculate_image_metrics(stego_image, compressed_stego)
    metrics_cover_compressed = calculate_image_metrics(cover_image, compressed_stego)

    # 3. Jalankan Ekstraksi Pesan pada Citra Terkompresi
    extract_start = time.perf_counter()
    recovered_payload = None
    extract_info: Dict[str, Any] = {}

    if method_lower == "lsb":
        recovered_payload, extract_time, extract_info = extract_lsb(
            compressed_stego, expected_payload_len=payload_len
        )
    elif method_lower == "adaptive_lsb":
        operator = kwargs.get("operator", "sobel")
        recovered_payload, extract_time, extract_info = extract_adaptive_lsb(
            compressed_stego, operator=operator, expected_payload_len=payload_len
        )
    elif method_lower == "dct":
        coeff_pair_1 = kwargs.get("coeff_pair_1", (4, 3))
        coeff_pair_2 = kwargs.get("coeff_pair_2", (3, 4))
        recovered_payload, extract_time, extract_info = extract_dct(
            compressed_stego,
            coeff_pair_1=coeff_pair_1,
            coeff_pair_2=coeff_pair_2,
            expected_payload_len=payload_len
        )
    else:
        raise ValueError(f"Metode '{method}' tidak dikenali.")

    total_extract_time = time.perf_counter() - extract_start

    # 4. Evaluasi Metrik Pesan (BER dan ESR)
    msg_metrics = calculate_message_metrics(
        original_bytes=original_payload,
        recovered_bytes=recovered_payload
    )

    # Status keberhasilan ekstraksi utuh 100%
    is_success = bool(
        msg_metrics["extraction_success"] and extract_info.get("success", False)
    )

    result = {
        "method": method_lower,
        "jpeg_quality": quality,
        "payload_bytes": payload_len,
        "compressed_file_size_bytes": file_size,
        # Metrik Citra
        "mse_cover_stego": metrics_cover_stego["mse"],
        "psnr_cover_stego": metrics_cover_stego["psnr"],
        "ssim_cover_stego": metrics_cover_stego["ssim"],
        "mse_stego_compressed": metrics_stego_compressed["mse"],
        "psnr_stego_compressed": metrics_stego_compressed["psnr"],
        "ssim_stego_compressed": metrics_stego_compressed["ssim"],
        "mse_cover_compressed": metrics_cover_compressed["mse"],
        "psnr_cover_compressed": metrics_cover_compressed["psnr"],
        "ssim_cover_compressed": metrics_cover_compressed["ssim"],
        # Metrik Pesan
        "ber": msg_metrics["ber"],
        "extraction_success": is_success,
        "extraction_time_seconds": total_extract_time,
        "extract_info": extract_info
    }

    return result


if __name__ == "__main__":
    print("=" * 78)
    print("DEMO SIMULASI SERANGAN KOMPRESI JPEG (FASE 5 / MILESTONE 4)")
    print("=" * 78)

    cover_file = "data/cover/xray_sample_01.png"
    secret_file = "data/secret/sample_64bytes.txt"
    qualities = [95, 90, 85, 80, 75]

    print(f"\n[1] Memuat Citra Cover: {cover_file}")
    cover = load_cover_image(cover_file)

    print(f"[2] Memuat Pesan Rahasia: {secret_file}")
    secret = load_secret_message(secret_file)
    print(f"    - Panjang Pesan: {len(secret)} bytes ({len(secret)*8} bit)")

    # Siapkan stego images untuk ketiga metode
    print("\n[3] Melakukan Embedding Awal (LSB, Adaptive LSB, DCT)...")
    stego_lsb, _, _ = embed_lsb(cover, secret)
    stego_adaptive, _, _ = embed_adaptive_lsb(cover, secret)
    stego_dct, _, _ = embed_dct(cover, secret, quantization_factor=25.0)

    stego_dict = {
        "LSB": (stego_lsb, "lsb"),
        "Adaptive-LSB": (stego_adaptive, "adaptive_lsb"),
        "DCT": (stego_dct, "dct")
    }

    print("\n[4] Menjalankan Serangan Kompresi JPEG (Quality 95 -> 75):")
    print("-" * 78)
    print(f"{'Metode':<14} | {'JPEG Q':<6} | {'BER':<8} | {'ESR':<7} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Ukuran (KB)':<10}")
    print("-" * 78)

    for method_name, (stego_img, method_code) in stego_dict.items():
        for q in qualities:
            out_file = f"outputs/compressed/stego_{method_code}_q{q}.jpg"
            res = evaluate_jpeg_attack(
                cover_image=cover,
                stego_image=stego_img,
                method=method_code,
                original_payload=secret,
                quality=q,
                compressed_output_path=out_file
            )
            esr_str = "BERHASIL" if res["extraction_success"] else "GAGAL"
            file_kb = res["compressed_file_size_bytes"] / 1024.0
            print(
                f"{method_name:<14} | Q={q:<4} | {res['ber']:<8.4f} | {esr_str:<7} | "
                f"{res['psnr_stego_compressed']:<10.2f} | {res['ssim_stego_compressed']:<8.4f} | {file_kb:<10.1f}"
            )
        print("-" * 78)

    print("\nDemo Simulasi Serangan JPEG selesai. File terkompresi tersimpan di outputs/compressed/.\n")

"""
Modul Orkestrator Eksperimen dan Ekspor Data CSV (Fase 6 / Milestone 4).
Memenuhi Persyaratan FR-07, FR-11, FR-14, FR-15, NFR-01, AC-08, AC-09, AC-10 PRD.md / STEP.md.

Modul ini mengotomatiskan matriks eksperimen steganografi:
Dataset Citra X-Ray x Metode (LSB, Adaptive LSB, DCT) x Ukuran Payload x Kualitas JPEG (termasuk Baseline tanpa kompresi).
Menghitung 3 pasangan metrik kualitas citra (Cover vs Stego, Stego vs Stego Terkompresi, Cover vs Stego Terkompresi),
mengukur waktu komputasi (Embedding & Extraction Time), serta mencatat kegagalan secara eksplisit ke dalam
file tabular CSV 'results/experiment_results.csv'.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import yaml

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
    from .jpeg_attack import compress_jpeg, evaluate_jpeg_attack
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
    from jpeg_attack import compress_jpeg, evaluate_jpeg_attack


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ExperimentRunner")


REQUIRED_CSV_COLUMNS = [
    "experiment_id",
    "image_id",
    "method",
    "payload_bytes",
    "payload_bpp",
    "jpeg_quality",
    "mse_cover_stego",
    "psnr_cover_stego",
    "ssim_cover_stego",
    "mse_stego_compressed",
    "psnr_stego_compressed",
    "ssim_stego_compressed",
    "ber",
    "extraction_success",
    "embedding_time_seconds",
    "extraction_time_seconds"
]


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Memuat file konfigurasi eksperimen YAML.
    
    Args:
        config_path: Jalur ke file config.yaml.
        
    Returns:
        Dict[str, Any]: Kamus konfigurasi eksperimen.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(
    config_path: str = "config.yaml",
    custom_config: Optional[Dict[str, Any]] = None,
    save_csv: bool = True,
    save_stego_images: bool = True
) -> pd.DataFrame:
    """
    Menjalankan seluruh rangkaian eksperimen otomatis berdasarkan konfigurasi.

    Args:
        config_path: Path file config.yaml.
        custom_config: Opsional, dictionary override untuk konfigurasi eksperimen.
        save_csv: Jika True, menyimpan hasil ke file CSV.
        save_stego_images: Jika True, menyimpan citra stego ke disk.

    Returns:
        pd.DataFrame: Dataframe berisi seluruh baris hasil eksperimen.
    """
    config = custom_config if custom_config is not None else load_config(config_path)

    # 1. Parsing konfigurasi
    exp_cfg = config.get("experiment", {})
    paths_cfg = config.get("paths", {})
    payload_cfg = config.get("payload_settings", {})
    jpeg_cfg = config.get("jpeg_attack_settings", {})
    dct_cfg = config.get("dct_settings", {})
    adapt_cfg = config.get("adaptive_lsb_settings", {})

    random_seed = exp_cfg.get("random_seed", 42)
    np.random.seed(random_seed)

    cover_dir = paths_cfg.get("cover_dir", "data/cover")
    secret_dir = paths_cfg.get("secret_dir", "data/secret")
    stego_dir = paths_cfg.get("stego_dir", "outputs/stego")
    compressed_dir = paths_cfg.get("compressed_dir", "outputs/compressed")
    recovered_dir = paths_cfg.get("recovered_dir", "outputs/recovered")
    results_csv = paths_cfg.get("results_csv", "results/experiment_results.csv")

    for d in [stego_dir, compressed_dir, recovered_dir, os.path.dirname(results_csv)]:
        if d:
            os.makedirs(d, exist_ok=True)

    methods = exp_cfg.get("methods", ["lsb", "adaptive_lsb", "dct"])
    payload_sizes = payload_cfg.get("payload_sizes_bytes", [64, 128, 256, 512, 1024])
    jpeg_qualities = jpeg_cfg.get("qualities", [95, 90, 85, 80, 75])

    # Kumpulkan citra cover
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    if "cover_files_override" in config:
        cover_files = config["cover_files_override"]
    else:
        cover_files = [
            f for f in sorted(os.listdir(cover_dir))
            if f.lower().endswith(valid_extensions) and not f.startswith(".")
        ]

    if not cover_files:
        raise ValueError(f"Tidak ada file citra cover yang ditemukan di direktori {cover_dir}")

    logger.info(f"Memulai Orkestrasi Eksperimen Steganografi X-Ray")
    logger.info(f"Citra Cover ({len(cover_files)}): {cover_files}")
    logger.info(f"Metode ({len(methods)}): {methods}")
    logger.info(f"Ukuran Payload ({len(payload_sizes)}): {payload_sizes} bytes")
    logger.info(f"Kualitas JPEG Serangan ({len(jpeg_qualities)}): {jpeg_qualities}")

    results_rows: List[Dict[str, Any]] = []
    exp_counter = 1

    # 2. Iterasi Matriks Eksperimen
    for cover_file in cover_files:
        cover_path = os.path.join(cover_dir, cover_file)
        image_id = os.path.splitext(cover_file)[0]
        cover_img = load_cover_image(cover_path)
        h, w = cover_img.shape

        for method in methods:
            for p_size in payload_sizes:
                secret_path = os.path.join(secret_dir, f"sample_{p_size}bytes.txt")
                if not os.path.isfile(secret_path):
                    logger.warning(f"File pesan rahasia {secret_path} tidak ditemukan. Dilewati.")
                    continue

                secret_bytes = load_secret_message(secret_path)
                bpp = calculate_bpp(p_size, (h, w))

                # Periksa kapasitas awal
                is_sufficient, req_bits, max_bits = check_capacity(
                    (h, w), p_size, method=method, overhead_bytes=4
                )

                if not is_sufficient:
                    # Catat penolakan kapasitas secara eksplisit untuk baseline dan semua kualitas
                    logger.info(
                        f"[{exp_counter:04d}] {image_id} | {method:<12} | {p_size}B | "
                        f"Kapasitas Tidak Mencukupi ({req_bits} bit > {max_bits} bit)"
                    )
                    error_msg = f"Payload {p_size} byte melebihi kapasitas maksimum ({max_bits // 8} byte)"

                    # Catat baris baseline (none)
                    row_base = {
                        "experiment_id": f"EXP-{exp_counter:04d}",
                        "image_id": image_id,
                        "method": method,
                        "payload_bytes": p_size,
                        "payload_bpp": round(bpp, 6),
                        "jpeg_quality": "none",
                        "mse_cover_stego": np.nan,
                        "psnr_cover_stego": np.nan,
                        "ssim_cover_stego": np.nan,
                        "mse_stego_compressed": np.nan,
                        "psnr_stego_compressed": np.nan,
                        "ssim_stego_compressed": np.nan,
                        "mse_cover_compressed": np.nan,
                        "psnr_cover_compressed": np.nan,
                        "ssim_cover_compressed": np.nan,
                        "ber": 1.0,
                        "extraction_success": False,
                        "embedding_time_seconds": 0.0,
                        "extraction_time_seconds": 0.0,
                        "status": "CAPACITY_EXCEEDED",
                        "error_message": error_msg
                    }
                    results_rows.append(row_base)
                    exp_counter += 1

                    # Catat baris kualitas JPEG
                    for q in jpeg_qualities:
                        row_q = row_base.copy()
                        row_q["experiment_id"] = f"EXP-{exp_counter:04d}"
                        row_q["jpeg_quality"] = str(q)
                        results_rows.append(row_q)
                        exp_counter += 1
                    continue

                # Lakukan Embedding
                stego_img = None
                embed_time = 0.0
                try:
                    if method == "lsb":
                        stego_img, embed_time, _ = embed_lsb(cover_img, secret_bytes)
                    elif method == "adaptive_lsb":
                        operator = adapt_cfg.get("gradient_operator", "sobel")
                        stego_img, embed_time, _ = embed_adaptive_lsb(
                            cover_img, secret_bytes, operator=operator
                        )
                    elif method == "dct":
                        q_factor = float(dct_cfg.get("quantization_factor", 25.0))
                        c1 = tuple(dct_cfg.get("coeff_pair_1", [4, 3]))
                        c2 = tuple(dct_cfg.get("coeff_pair_2", [3, 4]))
                        stego_img, embed_time, _ = embed_dct(
                            cover_img, secret_bytes,
                            quantization_factor=q_factor,
                            coeff_pair_1=c1, coeff_pair_2=c2
                        )
                except Exception as e:
                    logger.error(f"Error saat embedding {method} {p_size}B: {str(e)}")
                    continue

                # Simpan stego image jika diaktifkan
                stego_filename = f"stego_{image_id}_{method}_{p_size}bytes.png"
                stego_filepath = os.path.join(stego_dir, stego_filename)
                if save_stego_images:
                    save_image(stego_img, stego_filepath)

                # Hitung distorsi Cover vs Stego
                cov_stego_metrics = calculate_image_metrics(cover_img, stego_img)

                # --- 1. Skenario A: Baseline (Tanpa Serangan JPEG) ---
                rec_start = time.perf_counter()
                rec_payload = None
                extract_success = False

                if method == "lsb":
                    rec_payload, extract_time, ext_info = extract_lsb(stego_img)
                elif method == "adaptive_lsb":
                    rec_payload, extract_time, ext_info = extract_adaptive_lsb(
                        stego_img, operator=adapt_cfg.get("gradient_operator", "sobel")
                    )
                elif method == "dct":
                    rec_payload, extract_time, ext_info = extract_dct(
                        stego_img,
                        coeff_pair_1=tuple(dct_cfg.get("coeff_pair_1", [4, 3])),
                        coeff_pair_2=tuple(dct_cfg.get("coeff_pair_2", [3, 4]))
                    )

                base_msg_met = calculate_message_metrics(secret_bytes, rec_payload)
                base_esr = bool(base_msg_met["extraction_success"])

                baseline_row = {
                    "experiment_id": f"EXP-{exp_counter:04d}",
                    "image_id": image_id,
                    "method": method,
                    "payload_bytes": p_size,
                    "payload_bpp": round(bpp, 6),
                    "jpeg_quality": "none",
                    "mse_cover_stego": cov_stego_metrics["mse"],
                    "psnr_cover_stego": cov_stego_metrics["psnr"],
                    "ssim_cover_stego": cov_stego_metrics["ssim"],
                    "mse_stego_compressed": 0.0,
                    "psnr_stego_compressed": float("inf"),
                    "ssim_stego_compressed": 1.0,
                    "mse_cover_compressed": cov_stego_metrics["mse"],
                    "psnr_cover_compressed": cov_stego_metrics["psnr"],
                    "ssim_cover_compressed": cov_stego_metrics["ssim"],
                    "ber": base_msg_met["ber"],
                    "extraction_success": base_esr,
                    "embedding_time_seconds": round(embed_time, 6),
                    "extraction_time_seconds": round(extract_time, 6),
                    "status": "SUCCESS" if base_esr else "FAILED",
                    "error_message": ""
                }
                results_rows.append(baseline_row)
                exp_counter += 1

                # --- 2. Skenario B: Serangan Kompresi JPEG ---
                for q in jpeg_qualities:
                    comp_filename = f"comp_{image_id}_{method}_{p_size}bytes_q{q}.jpg"
                    comp_filepath = os.path.join(compressed_dir, comp_filename)

                    attack_res = evaluate_jpeg_attack(
                        cover_image=cover_img,
                        stego_image=stego_img,
                        method=method,
                        original_payload=secret_bytes,
                        quality=q,
                        compressed_output_path=comp_filepath,
                        operator=adapt_cfg.get("gradient_operator", "sobel"),
                        coeff_pair_1=tuple(dct_cfg.get("coeff_pair_1", [4, 3])),
                        coeff_pair_2=tuple(dct_cfg.get("coeff_pair_2", [3, 4]))
                    )

                    q_row = {
                        "experiment_id": f"EXP-{exp_counter:04d}",
                        "image_id": image_id,
                        "method": method,
                        "payload_bytes": p_size,
                        "payload_bpp": round(bpp, 6),
                        "jpeg_quality": str(q),
                        "mse_cover_stego": attack_res["mse_cover_stego"],
                        "psnr_cover_stego": attack_res["psnr_cover_stego"],
                        "ssim_cover_stego": attack_res["ssim_cover_stego"],
                        "mse_stego_compressed": attack_res["mse_stego_compressed"],
                        "psnr_stego_compressed": attack_res["psnr_stego_compressed"],
                        "ssim_stego_compressed": attack_res["ssim_stego_compressed"],
                        "mse_cover_compressed": attack_res["mse_cover_compressed"],
                        "psnr_cover_compressed": attack_res["psnr_cover_compressed"],
                        "ssim_cover_compressed": attack_res["ssim_cover_compressed"],
                        "ber": attack_res["ber"],
                        "extraction_success": attack_res["extraction_success"],
                        "embedding_time_seconds": round(embed_time, 6),
                        "extraction_time_seconds": round(attack_res["extraction_time_seconds"], 6),
                        "status": "SUCCESS" if attack_res["extraction_success"] else "ATTACK_FAILED",
                        "error_message": attack_res.get("extract_info", {}).get("error", "")
                    }
                    results_rows.append(q_row)
                    exp_counter += 1

                logger.info(
                    f"Selesai: {image_id} | {method:<12} | {p_size:<4}B | "
                    f"Baseline PSNR: {cov_stego_metrics['psnr']:.2f} dB"
                )

    # 3. Bentuk DataFrame
    df = pd.DataFrame(results_rows)

    # Susun kolom agar 16 kolom wajib berada di urutan terdepan
    all_cols = REQUIRED_CSV_COLUMNS + [c for c in df.columns if c not in REQUIRED_CSV_COLUMNS]
    df = df[all_cols]

    # 4. Simpan ke CSV
    if save_csv and results_csv:
        df.to_csv(results_csv, index=False)
        logger.info(f"Seluruh hasil eksperimen ({len(df)} baris) berhasil disimpan ke: {results_csv}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="X-Ray Steganography Research Platform - Experiment Orchestrator"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Jalur file konfigurasi eksperimen (default: config.yaml)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Menjalankan pengujian cepat (smoke test) dengan 1 citra, 1 payload, 1 kualitas"
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Override jalur file penyimpanan CSV hasil eksperimen"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    if args.quick:
        logger.info("Mode QUICK (Smoke Test) diaktifkan.")
        config["payload_settings"]["payload_sizes_bytes"] = [64]
        config["jpeg_attack_settings"]["qualities"] = [90]
        # Batasi citra hanya 1 file pertama
        cover_dir = config["paths"]["cover_dir"]
        covers = [f for f in sorted(os.listdir(cover_dir)) if not f.startswith(".")]
        if covers:
            config["cover_files_override"] = [covers[0]]

    if args.output_csv:
        config["paths"]["results_csv"] = args.output_csv

    df_results = run_experiment(custom_config=config)
    print("\n" + "=" * 80)
    print("RINGKASAN EKSEKUSI EKSPERIMEN:")
    print("=" * 80)
    print(f"Total Baris Pengujian : {len(df_results)}")
    print(f"File Hasil CSV        : {config['paths']['results_csv']}")
    print("\nCuplikan 5 Baris Pertama:")
    print(df_results[REQUIRED_CSV_COLUMNS[:10]].head().to_string())
    print("=" * 80 + "\n")

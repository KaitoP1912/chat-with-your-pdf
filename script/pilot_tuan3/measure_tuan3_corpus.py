"""
measure_tuan3_corpus.py — Bước 7: đo thực nghiệm toàn bộ Trạm 2 trên corpus thật.

Chạy Bước 0 -> 6 tuần tự trên từng file trong corpus, đo:
  - Số chunk sinh ra: chunk_by_page (thường + bridge) vs chunk_fixed_size
  - Thời gian: đọc+chuẩn hóa PDF / chunking / tách từ+embedding+FAISS
  - RAM ước tính: số chunk x 768 chiều x 4 byte (float32)

Kết quả lưu ra: results/tuan3_pilot/tuan3_measurement_summary.csv
Log lỗi (nếu có file nào crash) lưu ra: results/tuan3_pilot/tuan3_measurement_errors.txt

Cách chạy (từ thư mục gốc project, đã activate venv):
    python script/pilot_tuan3/measure_tuan3_corpus.py
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VNCORENLP_DIR = str(PROJECT_ROOT / "vncorenlp_models")
sys.path.insert(0, str(PROJECT_ROOT))

from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.chunker import chunk_by_page, chunk_fixed_size, default_token_counter
from source.retrieval.vectorstore import embed_chunks

# 7 file PDF hợp lệ của Trạm 2 (đã loại scan_nd238_14tr.pdf bị Trạm 1 reject,
# và word_118-2025-qh15_28tr.docx vì Trạm 2 hiện chỉ xử lý PDF, .docx là phần mở rộng).
CORPUS_FILES = [
    "normal_hienphap_33tr.pdf",
    "normal_vinamilkbaocao2014_53tr.pdf",
    "normal_lichsudang_C1&2_60tr.pdf",
    "mixedscan_qcvn06_38tr.pdf",
    "oldenc_vni_118-2025-qh15_23tr.pdf",
    "oldenc_tcvn3_36-2024-qh15_53tr.pdf",
    "nodiacritic_118-2025-qh15_20tr.pdf",
]

CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
RESULTS_DIR = PROJECT_ROOT / "results" / "tuan3_pilot"

BYTES_PER_FLOAT32 = 4
EMBED_DIM = 768


def estimate_ram_mb(n_chunks: int) -> float:
    return (n_chunks * EMBED_DIM * BYTES_PER_FLOAT32) / (1024 * 1024)


def measure_one_file(file_path: Path, token_counter) -> dict:
    """Chạy Bước 0->6 trên 1 file, trả về dict số liệu đo được."""
    row = {"file": file_path.name}

    t0 = time.time()
    pages = build_clean_pages(str(file_path))
    t1 = time.time()
    row["total_pages"] = len(pages)
    row["time_ingest_s"] = round(t1 - t0, 2)

    page_chunks = chunk_by_page(pages, token_counter=token_counter)
    fixed_chunks = chunk_fixed_size(pages, token_counter=token_counter)
    t2 = time.time()
    n_bridge = sum(1 for c in page_chunks if c["is_bridge"])
    n_normal_page = len(page_chunks) - n_bridge
    row["chunk_by_page_normal"] = n_normal_page
    row["chunk_by_page_bridge"] = n_bridge
    row["chunk_by_page_total"] = len(page_chunks)
    row["chunk_fixed_size_total"] = len(fixed_chunks)
    row["time_chunking_s"] = round(t2 - t1, 2)

    # Đo embedding + index trên chunk_by_page (chiến lược đề xuất chính)
    vectors = embed_chunks(page_chunks, VNCORENLP_DIR)
    t3 = time.time()
    row["time_embed_index_s"] = round(t3 - t2, 2)
    row["ram_estimate_mb"] = round(estimate_ram_mb(len(page_chunks)), 2)

    row["time_total_s"] = round(t3 - t0, 2)
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "tuan3_measurement_summary.csv"
    error_path = RESULTS_DIR / "tuan3_measurement_errors.txt"

    print("Đang tải tokenizer (cần mạng lần đầu)...")
    token_counter = default_token_counter()

    rows = []
    errors = []

    for filename in CORPUS_FILES:
        file_path = CORPUS_DIR / filename
        print(f"\n--- {filename} ---")
        if not file_path.exists():
            msg = f"[BỎ QUA] Không tìm thấy file: {file_path}"
            print(msg)
            errors.append(msg)
            continue
        try:
            row = measure_one_file(file_path, token_counter)
            rows.append(row)
            print(f"  Trang: {row['total_pages']} | "
                  f"chunk_by_page: {row['chunk_by_page_total']} (bridge={row['chunk_by_page_bridge']}) | "
                  f"chunk_fixed_size: {row['chunk_fixed_size_total']} | "
                  f"Tổng thời gian: {row['time_total_s']}s")
        except Exception as exc:
            msg = f"[LỖI] {filename}: {exc}\n{traceback.format_exc()}"
            print(msg)
            errors.append(msg)

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nĐã lưu CSV: {csv_path}")

        # Tổng cộng toàn corpus
        total_pages = sum(r["total_pages"] for r in rows)
        total_page_chunks = sum(r["chunk_by_page_total"] for r in rows)
        total_bridge = sum(r["chunk_by_page_bridge"] for r in rows)
        total_fixed_chunks = sum(r["chunk_fixed_size_total"] for r in rows)
        total_time = sum(r["time_total_s"] for r in rows)
        total_ram = sum(r["ram_estimate_mb"] for r in rows)
        print("\n=== TỔNG CỘNG TOÀN CORPUS ===")
        print(f"Tổng số trang: {total_pages}")
        print(f"Tổng chunk_by_page: {total_page_chunks} (gồm {total_bridge} bridge chunk)")
        print(f"Tổng chunk_fixed_size: {total_fixed_chunks}")
        print(f"Tổng thời gian xử lý: {total_time:.2f}s")
        print(f"Tổng RAM ước tính (vector chunk_by_page): {total_ram:.2f} MB")

    if errors:
        with error_path.open("w", encoding="utf-8") as f:
            f.write("\n\n".join(errors))
        print(f"\nCó {len(errors)} lỗi/cảnh báo, xem chi tiết tại: {error_path}")


if __name__ == "__main__":
    main()
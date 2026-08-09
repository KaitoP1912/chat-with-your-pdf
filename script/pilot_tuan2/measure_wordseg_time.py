"""
measure_wordseg_time.py — Nhóm 4 (Tuần 2 - Fixed & Multi-run Supported)

Đo thời gian tách từ (word segmentation) bằng py_vncorenlp trên corpus.
Hỗ trợ đo lặp lại nhiều lần độc lập trên cùng file để lấy giá trị trung bình chuẩn xác.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "ingestion"))

from pdf_loader import load_pdf_pages  # noqa: E402

LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "tuan2_pilot", "wordseg_time_output.txt"
)


def load_segmenter(save_dir: str):
    """Khởi tạo py_vncorenlp segmenter."""
    try:
        import py_vncorenlp
    except ImportError as exc:
        raise SystemExit(
            "Chưa cài py_vncorenlp. Chạy: pip install py_vncorenlp\n"
            f"Chi tiết lỗi: {exc}"
        )

    os.makedirs(save_dir, exist_ok=True)
    try:
        model = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=os.path.abspath(save_dir)
        )
    except Exception as exc:
        raise SystemExit(
            "Không khởi tạo được py_vncorenlp — kiểm tra: "
            "(1) đã cài Java JDK và biến môi trường JAVA_HOME chưa, "
            "(2) có quyền ghi vào thư mục save_dir không.\n"
            f"Chi tiết lỗi: {exc}"
        )
    return model


def measure_file(model, file_path: str) -> dict:
    """Đo thời gian tách từ cho 1 file PDF, trả về dict kết quả."""
    pages = load_pdf_pages(file_path)
    total_pages = len(pages)

    start = time.perf_counter()
    total_chunks_segmented = 0
    for page in pages:
        if not page.raw_text.strip():
            continue
        segmented = model.word_segment(page.raw_text)
        total_chunks_segmented += len(segmented)
    elapsed = time.perf_counter() - start

    sec_per_page = elapsed / total_pages if total_pages else 0.0

    return {
        "file": os.path.basename(file_path),
        "total_pages": total_pages,
        "elapsed_sec": elapsed,
        "sec_per_page": sec_per_page,
        "segments_returned": total_chunks_segmented,
    }


def append_log(result: dict) -> float:
    """Nối log vào file và trả về điểm trung bình hiện tại."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = (
        f"{timestamp}\t{result['file']}\t{result['total_pages']} trang\t"
        f"{result['elapsed_sec']:.3f}s tổng\t{result['sec_per_page']:.4f} s/trang\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)

    prev_runs = []
    if os.path.isfile(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            for l in f:
                if result["file"] in l and "s/trang" in l:
                    try:
                        val = float(l.split("\t")[-1].replace(" s/trang", "").strip())
                        prev_runs.append(val)
                    except ValueError:
                        continue
    avg = sum(prev_runs) / len(prev_runs) if prev_runs else result["sec_per_page"]
    print(f"  -> Lần đo ghi nhận: {result['sec_per_page']:.4f} s/trang | Trung bình ({len(prev_runs)} lần): {avg:.4f} s/trang")
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Đường dẫn file PDF cần đo")
    parser.add_argument(
        "--save-dir", default="./vncorenlp_models", help="Thư mục lưu model py_vncorenlp"
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Số lần chạy lặp lại độc lập để lấy trung bình"
    )
    args = parser.parse_args()

    model = load_segmenter(args.save_dir)
    print(f"Đang tiến hành đo thời gian tách từ cho: {args.file} ({args.runs} lần chạy)...")

    for i in range(1, args.runs + 1):
        print(f"\n--- Chạy lần {i}/{args.runs} ---")
        result = measure_file(model, args.file)
        append_log(result)

    print(f"\nĐã hoàn thành! Toàn bộ log đo lường được lưu tại: {LOG_PATH}")


if __name__ == "__main__":
    main()
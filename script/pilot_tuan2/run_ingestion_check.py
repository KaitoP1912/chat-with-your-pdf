"""
run_ingestion_check.py — Nhóm 6 (Tuần 2 - Fixed & CSV Enhanced)

Tự động kiểm thử dây chuyền Trạm 1 trên toàn bộ Corpus PDF và xuất báo cáo chi tiết.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "ingestion"))

from pdf_loader import PDFLoadError, load_pdf_pages  # noqa: E402
from scan_detector import detect_scan, should_reject  # noqa: E402
from text_normalizer import normalize_page_text  # noqa: E402

DEFAULT_CORPUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "corpus"
)
OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "tuan2_pilot", "ingestion_check_summary.csv"
)


def process_file(file_path: str) -> dict:
    file_name = os.path.basename(file_path)
    row = {
        "file": file_name,
        "status": "OK",
        "total_pages": "",
        "doc_scan_status": "",
        "scan_pages": "",
        "empty_pages": "",
        "low_text_pages": "",
        "image_parse_error_pages": "",
        "detected_encoding": "",
        "encoding_confidence": "",
        "encoding_warning": "",  # Cột mới bổ sung ghi nhận ambiguous/cảnh báo
        "likely_missing_diacritics": "",
        "elapsed_sec": "",
        "error": "",
    }

    start = time.perf_counter()
    try:
        pages = load_pdf_pages(file_path)
        row["total_pages"] = len(pages)

        scan_result = detect_scan(pages)
        row["doc_scan_status"] = scan_result.doc_status.value
        row["scan_pages"] = ",".join(str(p) for p in scan_result.scan_pages) or "-"
        row["empty_pages"] = ",".join(str(p) for p in scan_result.empty_pages) or "-"
        row["low_text_pages"] = ",".join(str(p) for p in scan_result.low_text_pages) or "-"
        row["image_parse_error_pages"] = ",".join(str(p) for p in scan_result.image_parse_error_pages) or "-"

        if should_reject(scan_result):
            row["status"] = "REJECTED"
            row["detected_encoding"] = "N/A (bị từ chối)"
            row["encoding_confidence"] = "N/A"
            row["encoding_warning"] = "File scan 100%"
            row["likely_missing_diacritics"] = "N/A"
            row["error"] = "Toàn bộ tài liệu là scan (full_scan) — từ chối xử lý theo quy tắc."
            row["elapsed_sec"] = f"{time.perf_counter() - start:.2f}"
            return row

        # Lấy tối đa 5 trang sạch có chữ để đo mẫu confidence
        sample_pages = [p for p in pages if p.page_number not in set(scan_result.scan_pages + scan_result.empty_pages)]
        combined_text = " ".join(p.raw_text for p in sample_pages[:5])

        if combined_text.strip():
            norm_result = normalize_page_text(combined_text)
            row["detected_encoding"] = norm_result.detected_encoding or "unicode_chuẩn"
            row["encoding_confidence"] = f"{norm_result.encoding_confidence:.2f}"
            row["encoding_warning"] = norm_result.encoding_warning or "OK"
            row["likely_missing_diacritics"] = norm_result.likely_missing_diacritics
        else:
            row["detected_encoding"] = "N/A (không có trang text)"
            row["encoding_confidence"] = "N/A"

    except PDFLoadError as exc:
        row["status"] = "REJECTED"
        row["error"] = str(exc)
        row["detected_encoding"] = "N/A"
        row["encoding_confidence"] = "N/A"
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = str(exc)
        row["detected_encoding"] = "N/A"
        row["encoding_confidence"] = "N/A"
    finally:
        row["elapsed_sec"] = f"{time.perf_counter() - start:.2f}"

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args()

    pdf_files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.pdf")))
    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong: {args.corpus_dir}")
        return

    results = []
    for file_path in pdf_files:
        print(f"Đang xử lý: {os.path.basename(file_path)} ...")
        results.append(process_file(file_path))

    header = [
        "file", "status", "total_pages", "doc_scan_status", "scan_pages",
        "empty_pages", "low_text_pages", "image_parse_error_pages", "detected_encoding",
        "encoding_confidence", "encoding_warning", "likely_missing_diacritics", "elapsed_sec", "error",
    ]
    print("\n" + "\t".join(header))
    for r in results:
        print("\t".join(str(r.get(h, "")) for h in header))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nĐã ghi bảng tổng hợp vào: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
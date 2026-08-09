"""
xem_noi_dung_trang.py — Script phụ (chỉ đọc, không đụng pipeline chính)

In ra nội dung của 1 hoặc nhiều trang sau khi đi qua đủ 3 bước Trạm 1:
  Bước 1 (pdf_loader)      -> text thô vừa cắt ra từ trang
  Bước 2 (scan_detector)   -> trang này được phân loại là gì (TEXT/LOW_TEXT/SCAN/EMPTY)
  Bước 3 (text_normalizer) -> text sau khi làm sạch (xóa số trang rác, convert bảng mã...)

Cách dùng:
    python xem_noi_dung_trang.py "<file.pdf>" <trang_bắt_đầu> [trang_kết_thúc]

Ví dụ:
    python xem_noi_dung_trang.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 1        # chỉ trang 1
    python xem_noi_dung_trang.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 2 3      # trang 2 và 3
    python xem_noi_dung_trang.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 2 5      # từ trang 2 đến trang 5
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "ingestion"))

from pdf_loader import load_pdf_pages          # noqa: E402
from scan_detector import classify_page        # noqa: E402
from text_normalizer import normalize_page_text  # noqa: E402


def print_one_page(page) -> None:
    # ----- BƯỚC 1: pdf_loader -----
    print("\n" + "#" * 70)
    print(f"### TRANG {page.page_number}")
    print("#" * 70)

    print(f"\n{'='*70}")
    print(f"BƯỚC 1 — pdf_loader.py (text thô vừa cắt ra)")
    print("=" * 70)
    print(f"  Số ký tự: {page.char_count}")
    print(f"  Số ảnh: {page.image_count} | Số object vector: {page.vector_object_count}")
    print(f"  Nội dung thô:\n{'-'*70}\n{page.raw_text}\n{'-'*70}")

    # ----- BƯỚC 2: scan_detector -----
    result = classify_page(page)
    print(f"\n{'='*70}")
    print(f"BƯỚC 2 — scan_detector.py (phân loại trang)")
    print("=" * 70)
    print(f"  Trạng thái trang: {result.status.value.upper()}")
    if result.status.value in ("scan", "empty"):
        print(f"  -> Trang này sẽ bị usable_pages() LOẠI BỎ, không đưa sang Trạm 2.")
    else:
        print(f"  -> Trang này hợp lệ, sẽ được đưa tiếp sang bước làm sạch.")

    # ----- BƯỚC 3: text_normalizer -----
    norm = normalize_page_text(page.raw_text)
    print(f"\n{'='*70}")
    print(f"BƯỚC 3 — text_normalizer.py (text sau khi làm sạch)")
    print("=" * 70)
    print(f"  Bảng mã phát hiện: {norm.detected_encoding or 'unicode_chuẩn'} "
          f"(confidence={norm.encoding_confidence:.2f})")
    print(f"  Mất dấu?: {norm.likely_missing_diacritics}")
    print(f"  Nội dung sau chuẩn hóa:\n{'-'*70}\n{norm.normalized_text}\n{'-'*70}")


def main():
    if len(sys.argv) < 3:
        print('Cách dùng: python xem_noi_dung_trang.py "<file.pdf>" <trang_bắt_đầu> [trang_kết_thúc]')
        print('Ví dụ 1 trang : python xem_noi_dung_trang.py "data/corpus/x.pdf" 1')
        print('Ví dụ nhiều trang (2 và 3): python xem_noi_dung_trang.py "data/corpus/x.pdf" 2 3')
        print('Ví dụ khoảng trang (2 đến 5): python xem_noi_dung_trang.py "data/corpus/x.pdf" 2 5')
        sys.exit(1)

    file_path = sys.argv[1]
    start_page = int(sys.argv[2])
    # Nếu không truyền trang kết thúc -> chỉ xem đúng 1 trang start_page.
    # Nếu có truyền -> xem TOÀN BỘ các trang liên tiếp từ start_page đến end_page (bao gồm cả 2 đầu).
    end_page = int(sys.argv[3]) if len(sys.argv) >= 4 else start_page

    if end_page < start_page:
        print(f"Trang kết thúc ({end_page}) phải >= trang bắt đầu ({start_page}).")
        sys.exit(1)

    pages = load_pdf_pages(file_path)
    total_pages = len(pages)
    pages_by_number = {p.page_number: p for p in pages}

    target_page_numbers = list(range(start_page, end_page + 1))
    missing = [pn for pn in target_page_numbers if pn not in pages_by_number]
    if missing:
        print(f"File chỉ có {total_pages} trang, không tìm thấy trang: {missing}.")
        sys.exit(1)

    print(f"File: {os.path.basename(file_path)} | Tổng số trang: {total_pages}")
    print(f"Đang xem trang {start_page}"
          + (f" đến {end_page} ({len(target_page_numbers)} trang)" if end_page != start_page else ""))

    for pn in target_page_numbers:
        print_one_page(pages_by_number[pn])


if __name__ == "__main__":
    main()
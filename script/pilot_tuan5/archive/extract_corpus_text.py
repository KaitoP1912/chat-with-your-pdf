"""
extract_corpus_text.py — CHẠY CỤC BỘ TRÊN MÁY BẠN (VSCode), không thuộc phạm vi
deliverable Tuần 5 — chỉ là công cụ hỗ trợ soạn Test Set (Việc 1).

Mục đích: trích xuất text TỪNG TRANG của các file PDF corpus, dùng ĐÚNG cách
mà source/ingestion/pdf_loader.py của dự án đang làm (crop 2.5% mép trên/dưới,
extract_text(layout=False)) — để bạn có "ground truth" khớp với những gì hệ
thống RAG thực sự nhìn thấy, dùng soạn câu hỏi + xác minh answer_reference/
expected_page cho test_questions.json.

QUY ƯỚC SỐ TRANG (rất quan trọng): số trang trong output = thứ tự vật lý
trong PDF (trang 1 = pdf.pages[0], trang 2 = pdf.pages[1], ...), giống hệt
pdf_loader.py — KHÔNG phải số in trên trang (vì có thể có bìa không đánh số).
Khi soạn câu hỏi, expected_page phải dùng đúng số trang kiểu này.

Cài đặt (nếu máy bạn chưa có):
    pip install pdfplumber

Cách chạy (mỗi lần 1 file):
    python extract_corpus_text.py "duong/dan/normal_hienphap_33tr.pdf"
    python extract_corpus_text.py "duong/dan/normal_vinamilkbaocao2014_53tr.pdf"
    python extract_corpus_text.py "duong/dan/normal_lichsudang_C1&2_60tr.pdf"

Output: 1 file .txt cùng tên (vd normal_hienphap_33tr.txt) ở cùng thư mục
script này, định dạng:

    === Trang 1/33 | 512 ký tự ===
    <nội dung text trang 1>

    === Trang 2/33 | 890 ký tự ===
    <nội dung text trang 2>
    ...

File .txt này nhỏ (vài trăm KB), upload thẳng lên chat Claude được (không
bị giới hạn như file PDF gốc) — hoặc bạn có thể mở bằng VSCode để tự đọc
đối chiếu song song khi mình soạn câu hỏi.

LƯU Ý: với các trang bảng biểu/biểu đồ phức tạp (đặc biệt báo cáo tài chính
Vinamilk), text extraction có thể vẫn xáo trộn thứ tự số liệu — đây là hạn
chế ĐÃ BIẾT của dự án (xem dev_16, dev_18 trong context handoff). Với những
trang đó, nên cross-check thêm bằng cách tự nhìn trực tiếp PDF (hoặc dùng
Gemini đọc ảnh trang cụ thể) trước khi chốt answer_reference.
"""

from __future__ import annotations

import os
import sys

CROP_TOP_RATIO = 0.025
CROP_BOTTOM_RATIO = 0.025


def extract_page_text_safe(page) -> str:
    """Y hệt logic trong source/ingestion/pdf_loader.py — giữ đồng bộ nếu file đó đổi."""
    try:
        height = page.height
        crop_top = height * CROP_TOP_RATIO
        crop_bottom = height * (1 - CROP_BOTTOM_RATIO)
        crop_box = (page.bbox[0], crop_top, page.bbox[2], crop_bottom)
        cropped_page = page.crop(crop_box)
        text = cropped_page.extract_text(layout=False) or ""
    except Exception:
        text = page.extract_text(layout=False) or ""
    return text


def main() -> None:
    if len(sys.argv) < 2:
        print("Cách dùng: python extract_corpus_text.py <duong_dan_file.pdf>")
        sys.exit(1)

    try:
        import pdfplumber
    except ImportError:
        print("Thiếu thư viện pdfplumber. Cài bằng: pip install pdfplumber")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"Không tìm thấy file: {pdf_path}")
        sys.exit(1)

    out_path = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Đang trích xuất {total} trang từ '{os.path.basename(pdf_path)}'...")

        with open(out_path, "w", encoding="utf-8") as f:
            for idx, page in enumerate(pdf.pages, start=1):
                text = extract_page_text_safe(page)
                char_count = len(text.strip())
                f.write(f"=== Trang {idx}/{total} | {char_count} ký tự ===\n")
                f.write(text.strip() + "\n\n")
                if idx % 10 == 0 or idx == total:
                    print(f"  ... đã xử lý {idx}/{total} trang")

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nXong. Đã ghi: {out_path} ({size_kb:.1f} KB)")
    print("Upload file .txt này lên chat (hoặc mở bằng VSCode để đối chiếu).")


if __name__ == "__main__":
    main()
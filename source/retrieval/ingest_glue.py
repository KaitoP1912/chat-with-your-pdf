"""
ingest_glue.py — Bước 0 của Trạm 2

Nối kết quả của 2 module Trạm 1 (Tuần 2) lại thành 1 danh sách trang "sạch",
sẵn sàng làm input cho chunker.py.

Vì sao cần file này:
  - pdf_loader.load_pdf_pages() trả về list[PageData]: có page_number, source_file,
    nhưng text (`raw_text`) CHƯA được chuẩn hóa bảng mã cũ / mất dấu.
  - text_normalizer.normalize_page_text() nhận raw_text, trả về NormalizationResult:
    có `normalized_text` (chữ sạch) nhưng KHÔNG có số trang.
  - Hai hàm này độc lập, chưa có sẵn chỗ nào nối chúng lại — build_clean_pages()
    làm đúng việc đó.
"""

from __future__ import annotations

from typing import List, TypedDict

from source.ingestion.pdf_loader import load_pdf_pages, PDFLoadError
from source.ingestion.text_normalizer import normalize_page_text


class CleanPage(TypedDict):
    page_number: int
    source_file: str
    text: str
    # Giữ lại 2 cờ hữu ích cho báo cáo/nghiệm thu, không bắt buộc dùng ở chunker
    likely_missing_diacritics: bool
    encoding_decision: str


def build_clean_pages(file_path: str) -> List[CleanPage]:
    """Trả về list các trang đã có đủ (page_number, source_file, text sạch).

    Raise PDFLoadError nếu file vượt giới hạn/hỏng (đúng hành vi của Trạm 1).
    """
    raw_pages = load_pdf_pages(file_path)  # list[PageData]

    clean_pages: List[CleanPage] = []
    for page in raw_pages:
        result = normalize_page_text(page.raw_text)
        clean_pages.append(
            CleanPage(
                page_number=page.page_number,
                source_file=page.source_file,
                text=result.normalized_text,
                likely_missing_diacritics=result.likely_missing_diacritics,
                encoding_decision=result.encoding_decision,
            )
        )
    return clean_pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Cách dùng: python -m source.retrieval.ingest_glue <đường_dẫn_file.pdf>")
        sys.exit(1)

    pages = build_clean_pages(sys.argv[1])
    print(f"Tổng số trang: {len(pages)}")
    for p in pages[:3]:
        preview = p["text"][:80].replace("\n", " ")
        print(f"  Trang {p['page_number']} | encoding={p['encoding_decision']} | "
              f"mất dấu={p['likely_missing_diacritics']} | preview: {preview!r}")

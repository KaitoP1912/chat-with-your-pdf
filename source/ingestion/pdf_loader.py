"""
pdf_loader.py 

Trích xuất text theo từng trang từ file PDF, giữ nguyên số trang gốc
theo cấu trúc PDF, kèm metadata (page_number, source_file, total_pages, image_count).

Giới hạn vận hành:
  - Dung lượng file tối đa: 20 MB
  - Số trang tối đa: 60 trang
  - Crop an toàn 2.5% mép trên/dưới tránh mất chữ thật.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import pdfplumber

MAX_FILE_SIZE_MB = 20
MAX_PAGES = 60

# Thu hẹp crop xuống 2.5% để tránh cắt vào chữ nội dung
CROP_TOP_RATIO = 0.025
CROP_BOTTOM_RATIO = 0.025


class PDFLoadError(Exception):
    """Lỗi khi tải/đọc file PDF (vượt giới hạn, file hỏng, không mở được...)."""


@dataclass
class PageData:
    """Dữ liệu 1 trang đã trích xuất."""

    page_number: int          # Bắt đầu từ 1, khớp đúng số trang PDF gốc
    source_file: str          # Tên file gốc (không kèm đường dẫn)
    total_pages: int          # Tổng số trang của file
    raw_text: str             # Text thô trích được
    image_count: int = 0      # Số lượng đối tượng ảnh raster (Image XObject) có trong trang
    vector_object_count: int = 0  # Tổng số rects + lines + curves (phát hiện nội dung vector hóa: stamp/scan không phải raster)
    image_read_failed: bool = False  # True nếu không đọc/parse được danh sách ảnh của trang (nghi vấn scan lỗi)
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.raw_text.strip()) if self.raw_text else 0

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "source_file": self.source_file,
            "total_pages": self.total_pages,
            "raw_text": self.raw_text,
            "image_count": self.image_count,
            "vector_object_count": self.vector_object_count,
            "image_read_failed": self.image_read_failed,
            "char_count": self.char_count,
        }


def _check_file_size(file_path: str, max_size_mb: float) -> float:
    """Trả về dung lượng file (MB). Raise PDFLoadError nếu vượt giới hạn hoặc sai định dạng."""
    if not os.path.isfile(file_path):
        raise PDFLoadError(f"Không tìm thấy file: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise PDFLoadError(f"File '{os.path.basename(file_path)}' không phải là file định dạng PDF.")

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise PDFLoadError(
            f"File '{os.path.basename(file_path)}' có dung lượng {size_mb:.2f} MB, "
            f"vượt giới hạn cho phép ({max_size_mb} MB)."
        )
    return size_mb


def _extract_page_text_safe(page: pdfplumber.page.Page) -> str:
    """Trích xuất text an toàn, bộc lề nhẹ 2.5% đỉnh và đáy."""
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


def load_pdf_pages(
    file_path: str,
    max_pages: int = MAX_PAGES,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
) -> List[PageData]:
    """Trích xuất text theo từng trang của 1 file PDF."""
    _check_file_size(file_path, max_file_size_mb)

    source_file = os.path.basename(file_path)
    pages: List[PageData] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            if total_pages > max_pages:
                raise PDFLoadError(
                    f"File '{source_file}' có {total_pages} trang, "
                    f"vượt giới hạn cho phép ({max_pages} trang)."
                )

            if total_pages == 0:
                raise PDFLoadError(f"File '{source_file}' không có trang nào.")

            for idx, page in enumerate(pdf.pages, start=1):
                # Tách riêng try/except cho text và ảnh: một lỗi parse ảnh (thường gặp ở
                # trang scan có object ảnh hỏng/nén lạ) không được phép làm mất luôn text,
                # và ngược lại. Trước đây 2 việc này dùng chung 1 try/except nên khi
                # page.images ném exception, cả text lẫn image_count đều bị set về rỗng/0,
                # khiến trang scan lỗi bị lọt xuống nhãn EMPTY thay vì SCAN.
                try:
                    text = _extract_page_text_safe(page)
                except Exception as exc:
                    text = ""
                    print(f"[WARN] Không trích được text trang {idx}/{total_pages} của '{source_file}': {exc}")

                img_read_failed = False
                try:
                    img_count = len(page.images)
                except Exception as exc:
                    img_count = 0
                    img_read_failed = True
                    print(f"[WARN] Không đọc được danh sách ảnh trang {idx}/{total_pages} của '{source_file}': {exc}")

                # Đếm object vector (rects/lines/curves): phát hiện trang có nội dung
                # đồ họa/scan bị vector-hóa (path vẽ tay) thay vì raster Image XObject —
                # loại này không lọt vào page.images nhưng vẫn là trang "không có chữ để
                # dùng", không nên bị coi là trang trắng thật.
                try:
                    vector_count = len(page.rects) + len(page.lines) + len(page.curves)
                except Exception as exc:
                    vector_count = 0
                    print(f"[WARN] Không đọc được object vector trang {idx}/{total_pages} của '{source_file}': {exc}")

                pages.append(
                    PageData(
                        page_number=idx,
                        source_file=source_file,
                        total_pages=total_pages,
                        raw_text=text,
                        image_count=img_count,
                        vector_object_count=vector_count,
                        image_read_failed=img_read_failed,
                    )
                )

    except PDFLoadError:
        raise
    except Exception as exc:
        raise PDFLoadError(f"Không mở được file '{source_file}': {exc}") from exc

    return pages


def load_pdf_pages_safe(
    file_path: str,
    max_pages: int = MAX_PAGES,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
) -> Optional[List[PageData]]:
    try:
        return load_pdf_pages(file_path, max_pages, max_file_size_mb)
    except PDFLoadError as exc:
        print(f"[ERROR] {exc}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Cách dùng: python pdf_loader.py <đường_dẫn_file.pdf>")
        sys.exit(1)

    result = load_pdf_pages(sys.argv[1])
    print(f"Tổng số trang: {result[0].total_pages}")
    for p in result[:3]:
        preview = p.raw_text[:80].replace("\n", " ")
        print(f"  Trang {p.page_number}: {p.char_count} ký tự | {p.image_count} ảnh | preview: {preview!r}")
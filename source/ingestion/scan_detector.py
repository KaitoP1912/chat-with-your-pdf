"""
scan_detector.py 

Phát hiện trang/tài liệu PDF dạng scan (không có text layer).
Khắc phục lỗi gán nhãn SCAN nhầm cho các trang ít chữ (LOW_TEXT).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

try:
    from .pdf_loader import PageData
except ImportError:
    from pdf_loader import PageData  # type: ignore

MIN_CHARS_FULL_TEXT = 100
MIN_CHARS_LOW_TEXT = 20

# Ngưỡng số object vector (rects+lines+curves) để coi 1 trang "0 chữ, 0 ảnh raster"
# là trang có nội dung đồ họa/scan bị vector-hóa (SCAN) thay vì trang trắng thật (EMPTY).
# Căn cứ thực đo trên mixedscan_qcvn06_38tr.pdf:
#   - Trang scan thật (do người kiểm tra xác nhận trực quan): trang 1 = 668 curves + 2 rects,
#     trang 38 = 379 curves + 38 rects.
#   - Trang trắng thật (xác nhận trực quan): trang 3/7/9 = 0 rects, 0 lines, 0 curves.
# Ngưỡng 5 để chừa biên an toàn cho các trang có khung viền trang trí đơn giản (1-3 rects)
# nhưng thực chất vẫn là trang trắng — không nhầm sang SCAN. Cần re-validate ngưỡng này nếu
# corpus mở rộng sang nhiều nguồn scan khác (ví dụ scan tạo ít curve hơn/nhiều rect hơn).
MIN_VECTOR_OBJECTS_FOR_SCAN = 5


class PageScanStatus(str, Enum):
    TEXT = "text"              # Trang đầy đủ chữ
    LOW_TEXT = "low_text"      # Trang ít chữ
    SCAN = "scan"              # Trang scan: chữ < 20 VÀ (có đối tượng ảnh HOẶC không đọc được danh sách ảnh)
    EMPTY = "empty"            # Trang trắng thật: 0 chữ, 0 ảnh, VÀ đọc danh sách ảnh thành công (không lỗi parse)


class DocScanStatus(str, Enum):
    TEXT = "text"                # Tất cả các trang đều có text
    MIXED_SCAN = "mixed_scan"    # Một phần trang bị scan (vẫn xử lý các trang sạch)
    FULL_SCAN = "full_scan"      # 100% trang không trống bị scan -> Từ chối xử lý


@dataclass
class PageScanResult:
    page_number: int
    char_count: int
    image_count: int
    image_read_failed: bool
    status: PageScanStatus


@dataclass
class DocScanResult:
    source_file: str
    total_pages: int
    doc_status: DocScanStatus
    scan_pages: List[int]       # Danh sách số trang bị scan
    empty_pages: List[int]      # Danh sách số trang trắng
    low_text_pages: List[int]   # Danh sách số trang ít chữ
    image_parse_error_pages: List[int]  # Trang mà page.images ném lỗi parse (để audit/log riêng)
    page_results: List[PageScanResult]

    def summary(self) -> str:
        return (
            f"{self.source_file}: {self.doc_status.value} "
            f"({len(self.scan_pages)}/{self.total_pages} trang scan, "
            f"{len(self.empty_pages)} trang trống, "
            f"{len(self.low_text_pages)} trang ít chữ)"
        )


def classify_page(page: PageData) -> PageScanResult:
    """
    Phân loại trang dựa trên mật độ ký tự và đối tượng ảnh.
    Đã sửa logic: Chỉ đánh giá SCAN khi image_count > 0 và char_count < 20.
    """
    image_read_failed = getattr(page, "image_read_failed", False)

    if page.char_count >= MIN_CHARS_FULL_TEXT:
        status = PageScanStatus.TEXT
    elif page.char_count >= MIN_CHARS_LOW_TEXT:
        status = PageScanStatus.LOW_TEXT
    else:
        # Nhóm < 20 ký tự
        if page.image_count > 0:
            status = PageScanStatus.SCAN
        elif page.char_count == 0 and (
            image_read_failed or page.vector_object_count >= MIN_VECTOR_OBJECTS_FOR_SCAN
        ):
            # Không có chữ, không có ảnh raster, NHƯNG hoặc (a) không đọc được danh sách
            # ảnh (lỗi parse), hoặc (b) có nhiều object vector bất thường (rects/lines/curves)
            # => nội dung trang thực chất là hình ảnh/scan bị vector-hóa, không phải trắng
            # thật -> gán SCAN, không gán nhầm EMPTY.
            status = PageScanStatus.SCAN
        elif page.char_count == 0 and page.image_count == 0:
            status = PageScanStatus.EMPTY
        else:
            # Có từ 1 - 19 ký tự nhưng 0 ảnh (ví dụ: trang bìa/phân đoạn ít chữ) -> LOW_TEXT
            status = PageScanStatus.LOW_TEXT

    return PageScanResult(
        page_number=page.page_number,
        char_count=page.char_count,
        image_count=page.image_count,
        image_read_failed=image_read_failed,
        status=status,
    )


def detect_scan(pages: List[PageData]) -> DocScanResult:
    if not pages:
        raise ValueError("Danh sách trang rỗng, không thể đánh giá scan.")

    page_results = [classify_page(p) for p in pages]
    scan_pages = [r.page_number for r in page_results if r.status == PageScanStatus.SCAN]
    empty_pages = [r.page_number for r in page_results if r.status == PageScanStatus.EMPTY]
    low_text_pages = [r.page_number for r in page_results if r.status == PageScanStatus.LOW_TEXT]
    image_parse_error_pages = [r.page_number for r in page_results if r.image_read_failed]

    total_pages = len(page_results)
    non_empty_count = total_pages - len(empty_pages)

    if non_empty_count > 0 and len(scan_pages) == non_empty_count:
        doc_status = DocScanStatus.FULL_SCAN
    elif scan_pages:
        doc_status = DocScanStatus.MIXED_SCAN
    else:
        doc_status = DocScanStatus.TEXT

    return DocScanResult(
        source_file=pages[0].source_file,
        total_pages=total_pages,
        doc_status=doc_status,
        scan_pages=scan_pages,
        empty_pages=empty_pages,
        low_text_pages=low_text_pages,
        image_parse_error_pages=image_parse_error_pages,
        page_results=page_results,
    )


def should_reject(doc_result: DocScanResult) -> bool:
    return doc_result.doc_status == DocScanStatus.FULL_SCAN


def usable_pages(pages: List[PageData], doc_result: DocScanResult) -> List[PageData]:
    invalid_set = set(doc_result.scan_pages + doc_result.empty_pages)
    return [p for p in pages if p.page_number not in invalid_set]


if __name__ == "__main__":
    import sys
    from pdf_loader import load_pdf_pages  # type: ignore

    if len(sys.argv) < 2:
        print("Cách dùng: python scan_detector.py <đường_dẫn_file.pdf>")
        sys.exit(1)

    loaded_pages = load_pdf_pages(sys.argv[1])
    result = detect_scan(loaded_pages)
    print(result.summary())
    if result.scan_pages:
        print(f"  Trang bị scan: {result.scan_pages}")
    if result.empty_pages:
        print(f"  Trang trống: {result.empty_pages}")
    if result.low_text_pages:
        print(f"  Trang nghi ngờ (ít chữ): {result.low_text_pages}")
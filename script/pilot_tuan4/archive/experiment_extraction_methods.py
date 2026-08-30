"""
experiment_extraction_methods.py — Thử nghiệm ĐỘC LẬP, KHÔNG đụng vào pdf_loader.py

Mục đích: trang 7 của normal_vinamilkbaocao2014_53tr.pdf có nhiều bảng/biểu đồ
nằm sát nhau (cơ cấu sở hữu vốn, biểu đồ tăng trưởng...). Cách trích text mặc
định trong pdf_loader.py (extract_text(layout=False)) làm xáo trộn thứ tự,
khiến dev_16 bị lấy nhầm số liệu dù đã đưa đúng trang (xem oracle_context_results.csv).

Script này thử nhiều cách trích khác nhau trên CÙNG 1 trang, in ra để bạn tự mắt
so sánh xem cách nào giữ đúng cặp "nhãn — số liệu" (vd "SCIC" đi liền "45,06%")
tốt hơn cách mặc định. KHÔNG sửa gì trong source/, chỉ đọc PDF và in ra màn hình.

Cách chạy:
    python experiment_extraction_methods.py --pdf data/corpus/normal_vinamilkbaocao2014_53tr.pdf --page 7
"""

from __future__ import annotations

import argparse

import pdfplumber


def method_default(page: pdfplumber.page.Page) -> str:
    """Cách hiện tại trong pdf_loader.py: extract_text(layout=False)."""
    return page.extract_text(layout=False) or ""


def method_layout_true(page: pdfplumber.page.Page) -> str:
    """layout=True: cố giữ vị trí ký tự theo tọa độ gốc trên trang (giữ khoảng
    cách/cột thay vì dồn chữ lại), thường tốt hơn cho bảng/nhiều cột."""
    return page.extract_text(layout=True) or ""


def method_words_sorted(page: pdfplumber.page.Page) -> str:
    """Tự sắp lại các từ theo (dòng ~ toạ độ y, rồi tới toạ độ x) thay vì thứ tự
    pdfplumber trả về mặc định — đôi khi thứ tự gốc của object trong PDF không
    theo đúng thứ tự đọc mắt thường (đây là nguồn gây xáo trộn phổ biến)."""
    words = page.extract_words()
    if not words:
        return ""
    # Gom theo dòng: làm tròn 'top' để các từ cùng 1 dòng (chênh lệch nhỏ do font)
    # được xếp chung nhóm, rồi trong mỗi dòng sắp theo x0 (trái -> phải).
    LINE_TOLERANCE = 3
    words_sorted = sorted(words, key=lambda w: (round(w["top"] / LINE_TOLERANCE), w["x0"]))
    lines = []
    current_line = []
    current_key = None
    for w in words_sorted:
        key = round(w["top"] / LINE_TOLERANCE)
        if current_key is None or key == current_key:
            current_line.append(w["text"])
        else:
            lines.append(" ".join(current_line))
            current_line = [w["text"]]
        current_key = key
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def method_left_right_columns(page: pdfplumber.page.Page) -> str:
    """Nhiều báo cáo dạng infographic (như file Vinamilk này) chia trang thành
    2 nửa trái/phải độc lập về nội dung. Cắt riêng nửa trái và nửa phải, trích
    text từng nửa rồi ghép lại — tránh việc pdfplumber đọc lẫn dòng của 2 cột
    khác nhau vào chung 1 dòng."""
    width = page.width
    left = page.crop((0, 0, width / 2, page.height))
    right = page.crop((width / 2, 0, width, page.height))
    left_text = left.extract_text(layout=True) or ""
    right_text = right.extract_text(layout=True) or ""
    return f"--- NỬA TRÁI ---\n{left_text}\n\n--- NỬA PHẢI ---\n{right_text}"


def method_tables(page: pdfplumber.page.Page) -> str:
    """Thử xem pdfplumber có tự nhận diện được bảng nào trên trang không.
    Nếu bảng (vd bảng chỉ số kinh doanh) được nhận diện đúng dạng lưới, kết quả
    ở đây sẽ RÕ RÀNG hơn hẳn text thô — đáng dùng riêng cho các trang loại này."""
    tables = page.extract_tables()
    if not tables:
        return "(Không phát hiện bảng nào theo thuật toán mặc định của pdfplumber)"
    out = []
    for i, table in enumerate(tables, start=1):
        out.append(f"[Bảng {i}]")
        for row in table:
            out.append(" | ".join(cell or "" for cell in row))
    return "\n".join(out)


METHODS = {
    "1_default_hien_tai": method_default,
    "2_layout_true": method_layout_true,
    "3_words_sorted_theo_toa_do": method_words_sorted,
    "4_cat_nua_trai_phai": method_left_right_columns,
    "5_thu_nhan_dien_bang": method_tables,
}

# Các cụm từ cần xuất hiện ĐÚNG CẠNH NHAU để coi là trích đúng — dùng để tự động
# gợi ý (không thay việc bạn đọc bằng mắt) cách nào có khả năng đúng hơn.
CHECK_PAIRS = [
    ("SCIC", "45,06%"),
    ("nước ngoài", "49%"),
    ("trong nước", "5,94%"),
]


def _proximity_score(text: str, label: str, value: str, window: int = 40) -> bool:
    """Kiểm tra thô: value có xuất hiện trong khoảng `window` ký tự quanh label
    không — không hoàn hảo nhưng đủ để gợi ý nhanh cách nào đáng xem kỹ hơn."""
    lowered = text.lower()
    label_pos = lowered.find(label.lower())
    if label_pos == -1:
        return False
    start = max(0, label_pos - window)
    end = min(len(text), label_pos + len(label) + window)
    return value in text[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Đường dẫn file PDF")
    parser.add_argument("--page", type=int, required=True, help="Số trang (bắt đầu từ 1)")
    args = parser.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        idx = args.page - 1
        if idx < 0 or idx >= len(pdf.pages):
            print(f"Trang {args.page} không tồn tại (file có {len(pdf.pages)} trang).")
            return
        page = pdf.pages[idx]

        for name, fn in METHODS.items():
            print("=" * 70)
            print(f"CÁCH: {name}")
            print("=" * 70)
            try:
                result = fn(page)
            except Exception as exc:  # noqa: BLE001 — chỉ để demo, in lỗi ra xem thử
                print(f"[LỖI khi chạy phương pháp này: {exc}]")
                continue
            print(result)

            print("\n--- Gợi ý nhanh (không thay việc đọc bằng mắt) ---")
            for label, value in CHECK_PAIRS:
                ok = _proximity_score(result, label, value)
                mark = "✅ gần nhau" if ok else "❌ không thấy gần nhau"
                print(f"  '{label}' gần '{value}': {mark}")
            print()


if __name__ == "__main__":
    main()
"""
diagnose_char_spacing.py — Script chẩn đoán (chỉ đọc, KHÔNG sửa gì trong pipeline)

Mục đích: Kiểm tra xem pdfplumber.extract_text() có đang chèn nhầm dấu cách giữa
2 ký tự liền kề (ví dụ giữa "đ" và "ờng" trong "đường") do khoảng cách toạ độ (kerning)
giữa chúng lớn hơn ngưỡng mặc định (x_tolerance) hay không, và thử vài giá trị
x_tolerance khác nhau để tìm giá trị phù hợp.

Cách dùng:
    python diagnose_char_spacing.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf" 2
"""
import sys
import pdfplumber


def main():
    if len(sys.argv) < 3:
        print('Cách dùng: python diagnose_char_spacing.py "<file.pdf>" <số_trang>')
        sys.exit(1)

    file_path = sys.argv[1]
    page_number = int(sys.argv[2])

    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[page_number - 1]

        # --- Phần A: soi tọa độ từng ký tự quanh chữ "đ"/"®" để đo khoảng cách thật ---
        print("=" * 70)
        print("PHẦN A — Khoảng cách (gap) giữa mỗi ký tự và ký tự liền sau nó")
        print("=" * 70)
        chars = page.chars
        target_chars = {"®", "đ", "Đ", "§"}  # các glyph đại diện cho "đ" ở TCVN3/Unicode
        count_shown = 0
        for i, ch in enumerate(chars[:-1]):
            if ch["text"] in target_chars:
                nxt = chars[i + 1]
                gap = nxt["x0"] - ch["x1"]
                context = "".join(c["text"] for c in chars[max(0, i - 2): i + 6])
                print(f"  Ký tự '{ch['text']}' (x1={ch['x1']:.2f}) -> "
                      f"'{nxt['text']}' (x0={nxt['x0']:.2f}) | gap = {gap:.2f} | context: {context!r}")
                count_shown += 1
                if count_shown >= 15:
                    break
        if count_shown == 0:
            print("  (Không tìm thấy ký tự 'đ'/'®' nào trong page.chars — có thể trang này không có.)")

        # So sánh với gap trung bình giữa các ký tự BÌNH THƯỜNG (không phải đ) để có baseline
        normal_gaps = []
        for i, ch in enumerate(chars[:-1]):
            if ch["text"] not in target_chars and ch["text"] != " ":
                nxt = chars[i + 1]
                if nxt["text"] != " ":
                    normal_gaps.append(nxt["x0"] - ch["x1"])
        if normal_gaps:
            avg_normal = sum(normal_gaps) / len(normal_gaps)
            print(f"\n  Gap trung bình giữa 2 ký tự thường (không phải đ, không phải space): {avg_normal:.3f}")

        # --- Phần B: thử lại extract_text() với nhiều x_tolerance khác nhau ---
        print("\n" + "=" * 70)
        print("PHẦN B — So sánh extract_text() với các giá trị x_tolerance khác nhau")
        print("=" * 70)
        for tol in [1, 1.5, 2, 3, 5, 8]:
            text = page.extract_text(layout=False, x_tolerance=tol) or ""
            # Đếm số lần xuất hiện pattern lỗi "đ" + space + (không phải khoảng trắng kép)
            broken_count = text.count("đ ") + text.count("® ")
            preview_idx = text.find("đ ")
            if preview_idx == -1:
                preview_idx = text.find("® ")
            preview = text[max(0, preview_idx - 10): preview_idx + 20] if preview_idx != -1 else "(không thấy pattern lỗi)"
            print(f"  x_tolerance={tol:<4} | số lần gặp 'đ '/'® ' (nghi lỗi): {broken_count:<4} | ví dụ: {preview!r}")


if __name__ == "__main__":
    main()
"""
diagnose_space_width.py — Script chẩn đoán (chỉ đọc, KHÔNG sửa gì trong pipeline)

Mục đích: So sánh ĐỘ RỘNG (width = x1 - x0) của:
  (a) các dấu cách "nghi ngờ" — đứng ngay sau các phụ âm hay gặp lỗi mất chữ "ư"
      (đ, ng, ph, tr, l, n...)
  (b) các dấu cách THẬT giữa 2 từ bình thường (để làm baseline so sánh)

Nếu (a) luôn có width ~0 (gần như không tồn tại) còn (b) có width rõ ràng (~2-4pt),
thì xác nhận: đây không phải dấu cách thật, mà là "chỗ trống" để lại do glyph "ư" bị
lỗi hoàn toàn trong font — có thể phát hiện tổng quát bằng ngưỡng width, không cần
liệt kê từng phụ âm đứng trước.

Cách dùng:
    python diagnose_space_width.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf" 2
"""
import sys
import pdfplumber


def main():
    if len(sys.argv) < 3:
        print('Cách dùng: python diagnose_space_width.py "<file.pdf>" <số_trang>')
        sys.exit(1)

    file_path = sys.argv[1]
    page_number = int(sys.argv[2])

    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[page_number - 1]
        chars = page.chars

        print("=" * 70)
        print("PHẦN A — Độ rộng (width) của TẤT CẢ ký tự space, kèm 2 ký tự trước/sau")
        print("=" * 70)

        space_widths = []
        for i, ch in enumerate(chars):
            if ch["text"] == " ":
                width = ch["x1"] - ch["x0"]
                space_widths.append(width)
                before = "".join(c["text"] for c in chars[max(0, i - 3): i])
                after = "".join(c["text"] for c in chars[i + 1: i + 4])
                flag = "  <-- NGHI VẤN (width ~0)" if width < 0.5 else ""
                print(f"  width={width:6.2f} | ...{before!r} [SPACE] {after!r}...{flag}")

        if space_widths:
            space_widths.sort()
            print(f"\n  Tổng số space trên trang: {len(space_widths)}")
            print(f"  Width nhỏ nhất: {min(space_widths):.2f} | lớn nhất: {max(space_widths):.2f}")
            near_zero = [w for w in space_widths if w < 0.5]
            normal = [w for w in space_widths if w >= 0.5]
            print(f"  Số space width < 0.5 (nghi vấn 'ư' bị mất): {len(near_zero)}")
            print(f"  Số space width >= 0.5 (space thật giữa 2 từ): {len(normal)}")
            if normal:
                print(f"  Width trung bình của space THẬT: {sum(normal)/len(normal):.2f}")
            if near_zero:
                print(f"  Width trung bình của space NGHI VẤN: {sum(near_zero)/len(near_zero):.2f}")


if __name__ == "__main__":
    main()
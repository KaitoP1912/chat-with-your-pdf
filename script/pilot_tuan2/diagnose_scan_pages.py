"""
diagnose_scan_pages.py — Script chẩn đoán (chỉ đọc, KHÔNG sửa gì trong pipeline)

Mục đích: In ra cấu trúc chi tiết của các trang nghi là scan nhưng bị pdfplumber
báo 0 ảnh (page.images rỗng), để xác định chính xác vì sao image_count = 0.

Cách dùng:
    python diagnose_scan_pages.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 1 3 7 9 38
"""
import sys
import pdfplumber


def dump_page_info(page, page_number: int) -> None:
    print(f"\n{'='*70}")
    print(f"TRANG {page_number}")
    print(f"{'='*70}")
    print(f"  Kích thước (mediabox / width x height): {page.width:.1f} x {page.height:.1f}")
    print(f"  Xoay (rotation): {getattr(page, 'rotation', 'N/A')}")

    text = page.extract_text() or ""
    print(f"  Số ký tự text (extract_text): {len(text.strip())}")
    if text.strip():
        print(f"  Preview text: {text.strip()[:100]!r}")

    print(f"  page.chars: {len(page.chars)}")
    print(f"  page.images: {len(page.images)}")
    for i, im in enumerate(page.images[:3]):
        print(f"    - image[{i}]: bbox={im.get('x0'),im.get('top'),im.get('x1'),im.get('bottom')} "
              f"size={im.get('width'),im.get('height')}")

    print(f"  page.rects: {len(page.rects)}")
    print(f"  page.lines: {len(page.lines)}")
    print(f"  page.curves: {len(page.curves)}")

    # Liệt kê object thô trong content stream để tìm XObject lồng nhau / inline image
    try:
        raw_objs = page.objects
        for key in raw_objs:
            print(f"  page.objects['{key}']: {len(raw_objs[key])}")
    except Exception as exc:
        print(f"  [Lỗi khi đọc page.objects: {exc}]")

    # Kiểm tra Form XObject khai báo trong Resources của trang (nếu có ảnh lồng bên trong)
    try:
        resources = page.page_obj.get("Resources") if hasattr(page, "page_obj") else None
        xobjects = None
        if resources:
            xobjects = resources.get("XObject")
        if xobjects:
            print(f"  Resources/XObject keys: {list(xobjects.keys())}")
            for name, xobj_ref in xobjects.items():
                try:
                    xobj = xobj_ref.resolve() if hasattr(xobj_ref, "resolve") else xobj_ref
                    subtype = xobj.get("Subtype")
                    print(f"    - XObject '{name}': Subtype={subtype}")
                except Exception as exc:
                    print(f"    - XObject '{name}': [không đọc được: {exc}]")
        else:
            print("  Resources/XObject: (không có hoặc rỗng)")
    except Exception as exc:
        print(f"  [Lỗi khi đọc Resources/XObject: {exc}]")


def main():
    if len(sys.argv) < 3:
        print('Cách dùng: python diagnose_scan_pages.py "<file.pdf>" <trang1> <trang2> ...')
        sys.exit(1)

    file_path = sys.argv[1]
    page_numbers = [int(x) for x in sys.argv[2:]]

    with pdfplumber.open(file_path) as pdf:
        for pn in page_numbers:
            if 1 <= pn <= len(pdf.pages):
                dump_page_info(pdf.pages[pn - 1], pn)
            else:
                print(f"Trang {pn} không tồn tại (file có {len(pdf.pages)} trang).")


if __name__ == "__main__":
    main()
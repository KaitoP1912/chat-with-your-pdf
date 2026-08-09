# check_all.py
import sys
import os

# Xác định đường dẫn thư mục gốc của dự án (nơi đang chạy terminal)
BASE_DIR = os.getcwd()
# Trỏ tới thư mục chứa pdf_loader.py và text_normalizer.py
INGESTION_DIR = os.path.join(BASE_DIR, "source", "ingestion")

# Thêm vào sys.path để Python tìm thấy module
sys.path.insert(0, INGESTION_DIR)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các hàm cần thiết
from source.ingestion.pdf_loader import load_pdf_pages
from source.ingestion.text_normalizer import normalize_page_text

def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python check_all.py <đường_dẫn_file.pdf> [tên_file_output.txt]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "check_tcvn3.txt"
    
    print(f"Đang tải và xử lý file: {pdf_path} ...")
    
    try:
        pages = load_pdf_pages(pdf_path)
    except Exception as e:
        print(f"Lỗi khi load PDF: {e}")
        sys.exit(1)
        
    if not pages:
        print("Không đọc được trang nào. Thoát.")
        sys.exit(1)

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in pages:
            # Gọi hàm chuẩn hóa text
            res = normalize_page_text(p.raw_text)
            
            f.write(f"######################################################################\n")
            f.write(f"### TRANG {p.page_number} / {p.total_pages} | Bảng mã: {res.detected_encoding} (conf={res.encoding_confidence:.2f})\n")
            f.write(f"######################################################################\n")
            f.write(f"{res.normalized_text}\n\n")
            
    print(f"✅ Xong! Đã xuất kết quả toàn bộ {len(pages)} trang ra file: {out_path}")

if __name__ == "__main__":
    main()
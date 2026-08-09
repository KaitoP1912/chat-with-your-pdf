"""
diagnose_encoding_chars.py

Script CHẨN ĐOÁN (không sửa dữ liệu) — liệt kê toàn bộ ký tự đặc biệt (non-ASCII)
xuất hiện trong raw_text trích từ PDF mã cũ (TCVN3/VNI), kèm tần suất, ngữ cảnh
VIẾT HOA hay thường, và vài từ ví dụ thật — để xây lại TCVN3_MAP / VNI_MAP chính
xác thay vì đoán.

Cách dùng: đặt file này CÙNG THƯ MỤC với pdf_loader.py (ví dụ source/ingestion/),
rồi chạy:
    python diagnose_encoding_chars.py <file1.pdf> [file2.pdf ...] [--max-examples N]

Ví dụ:
    python diagnose_encoding_chars.py ../../data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf
    python diagnose_encoding_chars.py ../../data/corpus/oldenc_vni_118-2025-qh15_23tr.pdf

Output: bảng ký tự — mã Unicode — tần suất — số lần xuất hiện trong từ VIẾT HOA
so với từ viết thường — vài từ ví dụ để đối chiếu tay.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

try:
    from .pdf_loader import load_pdf_pages
except ImportError:
    from pdf_loader import load_pdf_pages  # type: ignore

WORD_PATTERN = re.compile(r"\S+", re.UNICODE)


def is_all_caps_context(word: str) -> bool:
    """True nếu mọi chữ cái ASCII trong từ đều viết HOA (để phát hiện vấn đề case-loss)."""
    ascii_letters = [ch for ch in word if ch.isascii() and ch.isalpha()]
    if not ascii_letters:
        return False  # không đủ căn cứ ASCII để so sánh HOA/thường
    return all(ch.isupper() for ch in ascii_letters)


def analyze_file(file_path: str, max_examples: int = 5) -> None:
    pages = load_pdf_pages(file_path)
    full_text = "\n".join(p.raw_text for p in pages)

    char_freq: dict = defaultdict(int)
    char_examples: dict = defaultdict(list)
    char_caps_count: dict = defaultdict(int)
    char_lower_count: dict = defaultdict(int)

    for word in WORD_PATTERN.findall(full_text):
        special_chars = {ch for ch in word if not ch.isascii()}
        if not special_chars:
            continue
        caps_ctx = is_all_caps_context(word)
        for ch in special_chars:
            char_freq[ch] += 1
            if caps_ctx:
                char_caps_count[ch] += 1
            else:
                char_lower_count[ch] += 1
            if len(char_examples[ch]) < max_examples and word not in char_examples[ch]:
                char_examples[ch].append(word)

    print(f"\n=== {file_path} ===")
    print(f"Tổng số trang: {len(pages)} | Tổng số ký tự đặc biệt (unique): {len(char_freq)}\n")
    header = f"{'Ký tự':<8}{'Mã (U+)':<10}{'Tần suất':<10}{'Trong HOA':<12}{'Trong thường':<14}Ví dụ"
    print(header)
    print("-" * 110)

    for ch, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
        code_point = f"U+{ord(ch):04X}"
        examples = ", ".join(char_examples[ch])
        print(f"{ch!r:<8}{code_point:<10}{freq:<10}{char_caps_count[ch]:<12}{char_lower_count[ch]:<14}{examples}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python diagnose_encoding_chars.py <file1.pdf> [file2.pdf ...] [--max-examples N]")
        sys.exit(1)

    args = sys.argv[1:]
    max_ex = 5
    if "--max-examples" in args:
        idx = args.index("--max-examples")
        max_ex = int(args[idx + 1])
        del args[idx:idx + 2]

    for fp in args:
        analyze_file(fp, max_examples=max_ex)
"""
verify_noi_dung_thuc_v3.py — Giong ban 2, nhung tu ghi thang ra file bang
UTF-8 trong code (khong dung `> file.txt` cua PowerShell nua), de tranh
loi UnicodeEncodeError do console Windows mac dinh dung cp1252.

Cach dung:
    python verify_noi_dung_thuc_v3.py
(khong can them > gi ca, tu tao file ket_qua_verify.txt)
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber

CORPUS_DIR = Path('data/corpus')
OUT_PATH = Path('ket_qua_verify.txt')


def normalize(text: str) -> str:
    return re.sub(r'\s+', '', text)


def get_page_text(pdf_path: Path, page_number_1indexed: int) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        if page_number_1indexed < 1 or page_number_1indexed > len(pdf.pages):
            return ""
        page = pdf.pages[page_number_1indexed - 1]
        return page.extract_text() or ""


def main():
    questions = json.loads(Path('cau_hoi_moi_25.json').read_text(encoding='utf-8'))

    lines = []
    lines.append(f"Tong so cau kiem tra: {len(questions)}")
    lines.append("=" * 78)

    n_ok, n_fail, n_skip = 0, 0, 0

    for i, q in enumerate(questions, start=1):
        if not q.get('is_answerable'):
            n_skip += 1
            continue

        source = q['source_file']
        pages = q.get('expected_page')
        pages_list = pages if isinstance(pages, list) else [pages]
        ref = q.get('answer_reference', '')
        fingerprint = normalize(ref[:40])

        pdf_path = CORPUS_DIR / source
        found_on_any_page = False
        page_texts = {}

        for p in pages_list:
            if p is None:
                continue
            page_text = get_page_text(pdf_path, p)
            page_texts[p] = page_text
            if fingerprint and fingerprint in normalize(page_text):
                found_on_any_page = True
                break

        if found_on_any_page:
            n_ok += 1
            lines.append(f"[OK ]  Cau {i:2} ({source}, trang {pages_list})")
        else:
            n_fail += 1
            lines.append(f"[FAIL] Cau {i:2} ({source}, trang {pages_list})")
            lines.append(f"       Cau hoi: {q['question']}")
            lines.append(f"       answer_reference (60 ky tu dau can tim): {ref[:60]}...")
            for p, txt in page_texts.items():
                lines.append(f"       --- NOI DUNG THAT trang {p} (500 ky tu dau) ---")
                lines.append(f"       {txt[:500]}")
            lines.append("")

    lines.append("=" * 78)
    lines.append(f"KET QUA: {n_ok} OK | {n_fail} FAIL | {n_skip} bo qua (unanswerable)")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # In ra man hinh 1 dong xac nhan (dong nay khong co dau tieng Viet, an toan)
    print(f"Da ghi ket qua vao: {OUT_PATH.resolve()}")
    print(f"OK={n_ok} FAIL={n_fail} SKIP={n_skip}")


if __name__ == "__main__":
    main()
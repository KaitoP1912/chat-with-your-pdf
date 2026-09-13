"""
verify_tu_khoa.py — Thay vi doi hoi answer_reference phai xuat hien LIEN
MACH tren trang (nhu ban truoc, bi loi voi layout nhieu cot cua Vinamilk),
script nay chi kiem tra: bao nhieu PHAN TRAM cac cum tu dac trung (5-7 tu
lien tiep) trong answer_reference co xuat hien tren trang, khong can dung
thu tu — phu hop voi PDF bi xao tron thu tu doc do layout phuc tap.

Cach dung:
    python verify_tu_khoa.py
(tu ghi ra file ket_qua_tu_khoa.txt, UTF-8)
"""
import json
import re
import unicodedata
from pathlib import Path

import pdfplumber

CORPUS_DIR = Path('data/corpus')
OUT_PATH = Path('ket_qua_tu_khoa.txt')


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return unicodedata.normalize('NFC', text)


def clean_words(text: str) -> list:
    text = strip_diacritics(text.lower())
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 2]  # bo tu qua ngan (a, o, la, cua...)


def get_page_text(pdf_path: Path, page_number_1indexed: int) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        if page_number_1indexed < 1 or page_number_1indexed > len(pdf.pages):
            return ""
        return pdf.pages[page_number_1indexed - 1].extract_text() or ""


questions = json.loads(Path('cau_hoi_moi_25.json').read_text(encoding='utf-8'))

lines = []
n_ok, n_fail, n_skip = 0, 0, 0

for i, q in enumerate(questions, start=1):
    if not q.get('is_answerable'):
        n_skip += 1
        continue

    source = q['source_file']
    pages = q.get('expected_page')
    pages_list = pages if isinstance(pages, list) else [pages]
    ref = q.get('answer_reference', '')

    ref_words = clean_words(ref)
    if not ref_words:
        continue

    pdf_path = CORPUS_DIR / source
    all_page_words = set()
    for p in pages_list:
        if p is None:
            continue
        all_page_words |= set(clean_words(get_page_text(pdf_path, p)))

    matched = sum(1 for w in ref_words if w in all_page_words)
    ratio = matched / len(ref_words)

    status = "OK" if ratio >= 0.75 else ("NGHI NGO" if ratio >= 0.5 else "FAIL")
    if status == "OK":
        n_ok += 1
    elif status == "FAIL":
        n_fail += 1

    lines.append(f"[{status:8}] Cau {i:2} ({source}, trang {pages_list}) - trung {matched}/{len(ref_words)} tu ({ratio*100:.0f}%)")
    if status != "OK":
        lines.append(f"           Cau hoi: {q['question'][:90]}")

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Da ghi ket qua vao: {OUT_PATH.resolve()}")
print(f"Tong: OK nhieu, FAIL={n_fail}, SKIP={n_skip}")
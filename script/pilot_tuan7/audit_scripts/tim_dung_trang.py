"""
tim_dung_trang.py — Voi moi cau dang FAIL, quet TOAN BO cac trang trong
dung file PDF de tim trang THAT SU chua noi dung answer_reference. Co xu
ly rieng truong hop PDF bi rot dau thanh (nhu da phat hien o file Lich su
Dang) bang cach so sanh CA KIEU CO DAU LAN KIEU DA BO DAU.

Cach dung:
    python tim_dung_trang.py
(tu ghi ra file ket_qua_tim_trang.txt, UTF-8)
"""
import json
import re
import unicodedata
from pathlib import Path

import pdfplumber

CORPUS_DIR = Path('data/corpus')
OUT_PATH = Path('ket_qua_tim_trang.txt')

# Cau nao dang FAIL thi kiem tra lai (lay tu ket qua verify_noi_dung_thuc_v3
# truoc do). Neu muon quet lai TAT CA thi de rong list nay.
FAIL_QUESTION_SNIPPETS = [
    "Khảo sát 'Nơi làm việc tốt nhất",
    "các bên liên quan có thể phản ánh thông tin nhạy cảm",
    "Trách nhiệm của Vinamilk đối với người tiêu dùng",
    "Hội nghị Ban Chấp hành Trung ương Đảng họp từ ngày 14",
    "Tháng 6 năm 1925, Nguyễn Ái Quốc",
    "Bản dự thảo Đường lối cách mạng miền Nam",
    "Chiến dịch Hồ Chí Minh lịch sử giải phóng Sài Gòn",
    "Ủy ban thường vụ Quốc hội có quyền hạn gì khi Quốc hội",
    "Hội nghị lần thứ 11 và 12 của Ban Chấp hành Trung ương Đảng năm 1965",
]


def strip_diacritics(text: str) -> str:
    """Bo dau tieng Viet, ha ve chu cai co ban (a,e,i,o,u,d,...) de so sanh
    duoc voi PDF bi loi rot dau thanh."""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return unicodedata.normalize('NFC', text)


def normalize(text: str) -> str:
    return re.sub(r'\s+', '', text).lower()


def normalize_loose(text: str) -> str:
    """Chuan hoa manh: bo khoang trang + bo dau, dung khi nghi ngo PDF loi dau."""
    return strip_diacritics(normalize(text))


questions = json.loads(Path('cau_hoi_moi_25.json').read_text(encoding='utf-8'))

lines = []

for q in questions:
    if not any(snip in q['question'] for snip in FAIL_QUESTION_SNIPPETS):
        continue

    source = q['source_file']
    ref = q.get('answer_reference', '')
    fingerprint = ref[:50]
    fp_exact = normalize(fingerprint)
    fp_loose = normalize_loose(fingerprint)

    pdf_path = CORPUS_DIR / source

    lines.append("=" * 78)
    lines.append(f"CAU HOI: {q['question']}")
    lines.append(f"File: {source} | Trang dang ghi: {q.get('expected_page')}")
    lines.append(f"Dang tim: {fingerprint}...")
    lines.append("-" * 78)

    matches_exact = []
    matches_loose = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_norm = normalize(page_text)
            page_loose = normalize_loose(page_text)

            if fp_exact and fp_exact in page_norm:
                matches_exact.append(i)
            elif fp_loose and fp_loose in page_loose:
                matches_loose.append(i)

    if matches_exact:
        lines.append(f"=> TIM THAY KHOP CHINH XAC (co dau) o trang: {matches_exact}")
    if matches_loose:
        lines.append(f"=> TIM THAY KHOP SAU KHI BO DAU (nghi PDF loi rot dau) o trang: {matches_loose}")
    if not matches_exact and not matches_loose:
        lines.append("=> KHONG TIM THAY o BAT KY trang nao trong file nay.")
        lines.append("   -> Co the: cau tra loi bi Gemini bia/dien giai lai, khong phai trich nguyen van.")
        lines.append("   -> Can tu tim thu cong hoac bo cau nay, thay cau khac.")

    lines.append("")

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Da ghi ket qua vao: {OUT_PATH.resolve()}")
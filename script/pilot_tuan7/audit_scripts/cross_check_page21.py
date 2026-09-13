"""
cross_check_page21.py — Doi chung nguyen nhan loi mat dau o trang 21
(normal_lichsudang_C1&2_60tr.pdf), dung PyMuPDF (fitz) - mot thu vien doc
PDF hoan toan doc lap voi pdfplumber (engine giai ma font khac nhau).

Neu PyMuPDF cung cho ra loi mat dau y het -> loi nam trong file PDF that.
Neu PyMuPDF cho ra chu du dau, dung -> loi nam o cach pdfplumber giai ma
font cua file nay, khong phai loi nguon PDF.

Cai dat (neu chua co):
    pip install pymupdf --break-system-packages

Cach chay:
    python cross_check_page21.py
"""
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Chua cai PyMuPDF. Chay lenh sau roi thu lai:")
    print("  pip install pymupdf --break-system-packages")
    raise SystemExit(1)

PDF_PATH = "data/corpus/normal_lichsudang_C1&2_60tr.pdf"
PAGE_NUMBER = 21  # trang 21 (1-indexed, giong nhu xem_noi_dung_trang.py)

doc = fitz.open(PDF_PATH)
page = doc[PAGE_NUMBER - 1]  # fitz dung index tu 0
text = page.get_text()

print("=" * 70)
print(f"TRICH XUAT TRANG {PAGE_NUMBER} BANG PyMuPDF (fitz) - doc lap voi pdfplumber")
print("=" * 70)
print(text)
print("=" * 70)

# Kiem tra nhanh vai tu khoa da biet la bi loi trong ban pdfplumber
check_words = {
    "lưc": "lực",
    "cua": "của",
    "va ": "và ",
    "hanh": "hành",
    "thưc": "thực",
}
print("\nKiem tra nhanh cac tu da biet bi loi (ban pdfplumber):")
for wrong, correct in check_words.items():
    has_correct = correct in text
    has_wrong = wrong in text
    print(f"  '{correct}' (dung) co xuat hien? {has_correct}  |  "
          f"'{wrong}' (loi) co xuat hien? {has_wrong}")

doc.close()
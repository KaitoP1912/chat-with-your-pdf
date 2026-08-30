"""
extract_pages_for_relabel.py — trích nguyên văn PDF cho các trang cần xác minh

Mục đích: lấy ĐÚNG nguyên văn trang tài liệu gốc (chưa qua chunking/normalize)
để dán vào Gemini Pro làm bằng chứng, tránh việc Gemini tự suy đoán không có
căn cứ khi xác định câu hỏi answerable/partially_answerable/unanswerable.

Cách chạy (từ thư mục gốc dự án):
    python extract_pages_for_relabel.py
"""
import pdfplumber
from pathlib import Path

CORPUS_DIR = Path("data/corpus")

# (question_id, file, list các trang cần trích — lấy dư 1 trang trước/sau để
#  không bỏ sót nội dung vắt trang)
TARGETS = [
    ("dev_05", "normal_vinamilkbaocao2014_53tr.pdf", [6, 7, 8]),
    ("dev_06", "normal_vinamilkbaocao2014_53tr.pdf", [37, 38, 39]),
    ("dev_16", "normal_vinamilkbaocao2014_53tr.pdf", [6, 7, 8]),
    ("dev_18", "normal_vinamilkbaocao2014_53tr.pdf", [11, 12, 13]),
    ("dev_19", "normal_vinamilkbaocao2014_53tr.pdf", [27, 28, 29]),
    ("dev_10", "normal_lichsudang_C1&2_60tr.pdf", [10, 11, 12, 13]),
    ("dev_23", "normal_lichsudang_C1&2_60tr.pdf", [33, 34, 35, 36]),
    ("dev_25", "normal_lichsudang_C1&2_60tr.pdf", [39, 40, 41]),
]

out_path = Path("results/tuan4_pilot/oracle_page_texts.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as out:
    for qid, filename, pages in TARGETS:
        pdf_path = CORPUS_DIR / filename
        out.write(f"\n{'='*80}\n{qid} — {filename} — trang {pages}\n{'='*80}\n")
        with pdfplumber.open(pdf_path) as pdf:
            for p in pages:
                idx = p - 1  # pdfplumber 0-indexed
                if idx < 0 or idx >= len(pdf.pages):
                    out.write(f"\n[Trang {p}: KHÔNG TỒN TẠI trong file]\n")
                    continue
                text = pdf.pages[idx].extract_text() or "[Trang không trích được text]"
                out.write(f"\n--- Trang {p} ---\n{text}\n")
        print(f"Đã trích {qid}")

print(f"\nHoàn tất. Mở file: {out_path}")
print("Copy từng đoạn tương ứng dán vào prompt Gemini Pro theo mẫu đã gửi kèm.")
"""
Script đo đạc thống kê Corpus (Tuần 1 Pilot)
Bổ sung: Cảnh báo mật độ token/trang cao để dự báo nguy cơ nghẽn CPU ở khâu Embedding.
"""

import os
from datetime import datetime
import pdfplumber
from transformers import AutoTokenizer

PDF_PATH = "data/corpus/tests/normal_hienphap.pdf"
OUTPUT_PATH = "results/tuan1_pilot/corpus_stats_output.txt"

lines = []
def log(msg=""):
    """In ra màn hình đồng thời lưu vào buffer để ghi file."""
    print(msg)
    lines.append(str(msg))

log(f"KẾT QUẢ ĐO CORPUS STATS — chạy lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"File đo: {PDF_PATH}")
log("Đang tải tokenizer vietnamese-bi-encoder...")

tokenizer = AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-bi-encoder")

with pdfplumber.open(PDF_PATH) as pdf:
    total_pages = len(pdf.pages)
    total_tokens = 0
    page_token_counts = []
    empty_pages = 0

    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if len(text.strip()) == 0:
            empty_pages += 1
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        page_token_counts.append(n_tokens)
        total_tokens += n_tokens

avg_tokens_per_page = total_tokens / total_pages if total_pages > 0 else 0

log(f"\n--- KẾT QUẢ ĐO: {PDF_PATH} ---")
log(f"Tổng số trang: {total_pages}")
log(f"Số trang rỗng/không trích được text: {empty_pages}")
log(f"Tổng token toàn văn bản: {total_tokens:,}")
log(f"Trung bình token/trang: {avg_tokens_per_page:.1f}")
log(f"Trang ít token nhất: {min(page_token_counts) if page_token_counts else 0}")
log(f"Trang nhiều token nhất: {max(page_token_counts) if page_token_counts else 0}")
log(f"Số chunk 256-token ước tính (không tính overlap): {total_tokens // 256}")
log(f"Số trang cần để có ~10 chunk (đủ test bridge chunk): {round(10 * 256 / avg_tokens_per_page) if avg_tokens_per_page > 0 else 0}")

# --- ĐIỀU CHỈNH KỸ THUẬT: Cảnh báo mật độ chữ ---
if avg_tokens_per_page > 800:
    log("\n⚠️ CẢNH BÁO MẬT ĐỘ CHỮ DÀY:")
    log(f"  Văn bản có trung bình {avg_tokens_per_page:.1f} token/trang (>800).")
    log("  Khuyến nghị: Hạ giới hạn số trang upload xuống dưới 50 trang để không bị Timeout CPU ở khâu Ingestion.")

# Tự động ghi ra file (tạo thư mục nếu chưa có)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n💾 Đã lưu kết quả vào: {OUTPUT_PATH}")
"""
Đo thời gian embedding thật trên CPU — chạy trên file 114 trang (gần sát mốc 125 trang)
để xác nhận số liệu ngoại suy từ file 33 trang có còn đúng ở quy mô lớn hơn không.
Cùng cơ chế append như bản chạy trên normal_hienphap.pdf, nhưng lưu vào file riêng
để không lẫn với dữ liệu đo trên file 33 trang.
"""

import os
import re
import time
from datetime import datetime
import pdfplumber
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/corpus/tests/normal_lichsudang_114tr.pdf"
OUTPUT_PATH = "results/tuan1_pilot/embedding_time_output_114tr.txt"
TARGET_MAX_PAGES = 125
TIMEOUT_LIMIT_SEC = 90
WORDS_PER_CHUNK = 170

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

def append_log():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    separator = "\n" + ("=" * 70) + "\n"
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(separator + "\n".join(lines) + "\n")
    print(f"\n💾 Đã nối kết quả lần chạy này vào: {OUTPUT_PATH}")

def print_running_average():
    if not os.path.exists(OUTPUT_PATH):
        return
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        content = f.read()
    values = [float(v) for v in re.findall(r"Trung bình:\s*([\d.]+)\s*giây/trang", content)]
    if not values:
        return
    avg = sum(values) / len(values)
    log(f"\n--- TRUNG BÌNH TẤT CẢ {len(values)} LẦN CHẠY ĐÃ LƯU TRONG FILE NÀY (114 trang) ---")
    log(f"Các giá trị giây/trang từng lần: {values}")
    log(f"Trung bình: {avg:.3f} giây/trang")
    log(f"Ước tính cho {TARGET_MAX_PAGES} trang (trung bình {len(values)} lần, ngoại suy từ 114 trang): {avg * TARGET_MAX_PAGES:.2f} giây")

log(f"ĐO THỜI GIAN EMBEDDING (CPU) trên file 114 trang — chạy lúc {run_timestamp}")
log(f"File đo: {PDF_PATH}")

log("\n📄 Đang đọc PDF và chia chunk tạm thời...")
with pdfplumber.open(PDF_PATH) as pdf:
    total_pages = len(pdf.pages)
    all_text = [page.extract_text() or "" for page in pdf.pages]

chunks = []
for text in all_text:
    words = text.split()
    for i in range(0, len(words), WORDS_PER_CHUNK):
        chunk = " ".join(words[i:i + WORDS_PER_CHUNK])
        if chunk.strip():
            chunks.append(chunk)

log(f"✅ Tổng số trang: {total_pages}, số chunk tạo được: {len(chunks)}")

log("\n🧠 Đang tải model vietnamese-bi-encoder...")
model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device="cpu")

log(f"🚀 Đang encode {len(chunks)} chunk trên CPU...")
embed_start = time.time()
embeddings = model.encode(chunks, show_progress_bar=False)
embed_elapsed = time.time() - embed_start

sec_per_page = embed_elapsed / total_pages
projected_125_pages = sec_per_page * TARGET_MAX_PAGES

log(f"\n--- KẾT QUẢ ĐO EMBEDDING (114 trang) ---")
log(f"Số chunk đã encode: {len(chunks)}")
log(f"Thời gian encode toàn bộ ({total_pages} trang): {embed_elapsed:.2f} giây")
log(f"Trung bình: {sec_per_page:.3f} giây/trang")
log(f"Ước tính cho {TARGET_MAX_PAGES} trang (ngoại suy từ 114 trang): {projected_125_pages:.2f} giây")

log(f"\n--- ĐỐI CHIẾU VỚI TIMEOUT {TIMEOUT_LIMIT_SEC} GIÂY ---")
if projected_125_pages < TIMEOUT_LIMIT_SEC:
    log(f"✅ {projected_125_pages:.1f}s < {TIMEOUT_LIMIT_SEC}s.")
else:
    log(f"⚠️ {projected_125_pages:.1f}s ĐÃ VƯỢT {TIMEOUT_LIMIT_SEC}s -> cần xem lại giới hạn số trang.")

print_running_average()
append_log()
"""
Đo thời gian embedding thật trên CPU (Tuần 1 pilot)
Bản cập nhật:
- Đã hạ TARGET_MAX_PAGES từ 125 xuống 60 trang dựa trên thực nghiệm nghẽn CPU file 114 trang.
- Bổ sung Guardrail chặn ngay nếu PDF vượt quá 60 trang để bảo đảm Timeout < 90s.
"""

import os
import re
import time
from datetime import datetime
import pdfplumber
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/corpus/tests/normal_hienphap.pdf"
OUTPUT_PATH = "results/tuan1_pilot/embedding_time_output.txt"

# --- ĐIỀU CHỈNH KỸ THUẬT: Giới hạn số trang & Timeout mới ---
TARGET_MAX_PAGES = 60      # Khóa trần 60 trang cho bản MVP chạy CPU (được suy ra từ thực nghiệm)
TIMEOUT_LIMIT_SEC = 90     # Giới hạn timeout 90 giây
WORDS_PER_CHUNK = 170      # ~256 token

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

def append_log():
    """Nối khối kết quả của lần chạy này vào CUỐI file chung."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    separator = "\n" + ("=" * 70) + "\n"
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(separator + "\n".join(lines) + "\n")
    print(f"\n💾 Đã nối kết quả lần chạy này vào: {OUTPUT_PATH}")

def print_running_average():
    """Đọc lại toàn bộ file và tính trung bình thời gian xử lý."""
    if not os.path.exists(OUTPUT_PATH):
        return
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        content = f.read()
    values = [float(v) for v in re.findall(r"Trung bình:\s*([\d.]+)\s*giây/trang", content)]
    if not values:
        return
    avg = sum(values) / len(values)
    log(f"\n--- TRUNG BÌNH TẤT CẢ {len(values)} LẦN CHẠY ĐÃ LƯU TRONG FILE NÀY ---")
    log(f"Các giá trị giây/trang từng lần: {values}")
    log(f"Trung bình: {avg:.3f} giây/trang")
    log(f"Ước tính cho {TARGET_MAX_PAGES} trang: {avg * TARGET_MAX_PAGES:.2f} giây")

log(f"ĐO THỜI GIAN EMBEDDING (CPU) — chạy lúc {run_timestamp}")
log(f"File đo: {PDF_PATH}")

# --- Bước 1: Đọc PDF & Kiểm tra Guardrail Số Trang ---
log("\n📄 Đang đọc PDF...")
with pdfplumber.open(PDF_PATH) as pdf:
    total_pages = len(pdf.pages)
    
    # GUARDRAIL: Chặn ngay nếu file quá dài
    if total_pages > TARGET_MAX_PAGES:
        log(f"❌ LỖI VẬN HÀNH: File có {total_pages} trang, vượt quá giới hạn an toàn {TARGET_MAX_PAGES} trang!")
        log(f"   -> Đã từ chối xử lý để tránh vi phạm Timeout {TIMEOUT_LIMIT_SEC}s.")
        append_log()
        raise ValueError(f"File vượt quá giới hạn {TARGET_MAX_PAGES} trang cho phép.")

    all_text = [page.extract_text() or "" for page in pdf.pages]

chunks = []
for text in all_text:
    words = text.split()
    for i in range(0, len(words), WORDS_PER_CHUNK):
        chunk = " ".join(words[i:i + WORDS_PER_CHUNK])
        if chunk.strip():
            chunks.append(chunk)

log(f"✅ Tổng số trang: {total_pages}, số chunk tạo được: {len(chunks)}")

# --- Bước 2: Đo tách từ py_vncorenlp (Nếu có) ---
log("\n🔤 Đang thử đo thời gian tách từ (py_vncorenlp, chỉ đo 10 chunk đầu)...")
try:
    import py_vncorenlp
    model_dir = os.path.abspath("vncorenlp_models")
    os.makedirs(model_dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=model_dir)
    segmenter = py_vncorenlp.VnCoreNLP(save_dir=model_dir, annotators=["wseg"])
    seg_start = time.time()
    for chunk in chunks[:10]:
        segmenter.word_segment(chunk)
    seg_elapsed = time.time() - seg_start
    log(f"✅ Tách từ 10 chunk: {seg_elapsed:.2f}s (~{seg_elapsed/10:.3f}s/chunk)")
except Exception as e:
    log(f"⚠️ Bỏ qua đo tách từ (py_vncorenlp chưa sẵn sàng): {e}")

# --- Bước 3: Đo thời gian embedding thật trên CPU ---
log("\n🧠 Đang tải model vietnamese-bi-encoder...")
model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device="cpu")

log(f"🚀 Đang encode {len(chunks)} chunk trên CPU...")
embed_start = time.time()
embeddings = model.encode(chunks, show_progress_bar=False)
embed_elapsed = time.time() - embed_start

sec_per_page = embed_elapsed / total_pages
projected_target_pages = sec_per_page * TARGET_MAX_PAGES

log(f"\n--- KẾT QUẢ ĐO EMBEDDING ---")
log(f"Số chunk đã encode: {len(chunks)}")
log(f"Thời gian encode toàn bộ ({total_pages} trang): {embed_elapsed:.2f} giây")
log(f"Trung bình: {sec_per_page:.3f} giây/trang")
log(f"Ước tính cho {TARGET_MAX_PAGES} trang (chỉ riêng embedding): {projected_target_pages:.2f} giây")

log(f"\n--- ĐỐI CHIẾU VỚI TIMEOUT {TIMEOUT_LIMIT_SEC} GIÂY ---")
if projected_target_pages < TIMEOUT_LIMIT_SEC:
    log(f"✅ {projected_target_pages:.1f}s < {TIMEOUT_LIMIT_SEC}s -> Vẫn nằm trong ngưỡng thời gian an toàn.")
else:
    log(f"⚠️ {projected_target_pages:.1f}s ĐÃ VƯỢT {TIMEOUT_LIMIT_SEC}s -> Cần giảm giới hạn số trang hơn nữa!")

print_running_average()
append_log()
"""
Test baseline long-context trên file 114 trang (gần sát mốc 125 trang) — để xác nhận
số token/latency thực tế ở quy mô lớn hơn 33 trang, thay vì chỉ ngoại suy tuyến tính.
Câu hỏi dùng dạng tổng quát (tóm tắt + trích dẫn trang) vì mục đích chính là đo
token/latency, không phải chấm độ chính xác nội dung ở bước này.
"""

import os
import time
from datetime import datetime
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai

PDF_PATH = "data/corpus/tests/normal_lichsudang_114tr.pdf"
QUESTION = "Tóm tắt 3 ý chính của văn bản. Trả lời kèm số trang trích dẫn theo marker [Trang X] có trong văn bản."
MODEL_NAME = "gemini-flash-latest"
OUTPUT_PATH = "results/tuan1_pilot/longcontext_baseline_output_114tr.txt"

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

def save_log():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n💾 Đã lưu kết quả vào: {OUTPUT_PATH}")

log(f"KẾT QUẢ TEST BASELINE LONG-CONTEXT (114 trang) — chạy lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"File input: {PDF_PATH}")
log(f"Model: {MODEL_NAME}")
log(f"Câu hỏi: {QUESTION}")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or "dán_key" in api_key:
    log("❌ Chưa có GEMINI_API_KEY hợp lệ trong .env.")
    save_log()
    raise SystemExit("Kiểm tra lại .env trước khi chạy.")

genai.configure(api_key=api_key)

log(f"\n📄 Đang đọc {PDF_PATH} và chèn marker trang...")
with pdfplumber.open(PDF_PATH) as pdf:
    total_pages = len(pdf.pages)
    parts = []
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"[Trang {i}]\n{text}")
    full_text = "\n\n".join(parts)

char_count = len(full_text)
log(f"✅ Đã ghép {total_pages} trang, tổng {char_count:,} ký tự.")

model = genai.GenerativeModel(MODEL_NAME)
log("🔢 Đang đếm token qua Gemini API (count_tokens)...")
try:
    token_info = model.count_tokens(full_text)
    input_token_count = token_info.total_tokens
    log(f"✅ Số token input (ước tính trước): {input_token_count:,}")
except Exception as e:
    log(f"⚠️ Không đếm được token trước: {e}")
    input_token_count = None

prompt = f"""Bạn là trợ lý trả lời câu hỏi dựa trên văn bản dưới đây.
Chỉ dùng thông tin có trong văn bản, trích dẫn số trang theo marker [Trang X] xuất hiện trong văn bản.
Nếu không có thông tin, hãy nói rõ là không tìm thấy.

VĂN BẢN:
{full_text}

CÂU HỎI: {QUESTION}
"""

log(f"\n🚀 Đang gọi Gemini ({MODEL_NAME}) với toàn bộ {total_pages} trang...")
start = time.time()
try:
    response = model.generate_content(prompt)
    elapsed = time.time() - start

    log(f"\n✅ THÀNH CÔNG — Latency: {elapsed:.2f} giây")
    log("\n--- CÂU TRẢ LỜI ---")
    log(response.text)

    if hasattr(response, "usage_metadata"):
        um = response.usage_metadata
        log("\n--- THÔNG TIN TOKEN (từ response thật) ---")
        log(f"Prompt tokens: {um.prompt_token_count:,}")
        log(f"Output tokens: {um.candidates_token_count:,}")
        log(f"Tổng tokens: {um.total_token_count:,}")

except Exception as e:
    elapsed = time.time() - start
    log(f"\n❌ LỖI sau {elapsed:.2f} giây: {e}")
    log("Ghi chú: nếu lỗi 429/RESOURCE_EXHAUSTED -> đã chạm giới hạn free tier.")

log(f"\n--- TÓM TẮT ---")
log(f"Số trang: {total_pages}")
log(f"Số ký tự: {char_count:,}")
if input_token_count:
    log(f"Số token input (ước tính trước): {input_token_count:,}")
    log(f"Token/trang (114 trang): {input_token_count / total_pages:.1f}")

save_log()
"""
config.py — Bước A1 (kế hoạch "80% đồ án")

File này KHÔNG tính toán gì mới. Chỉ chép lại các quyết định đã chốt ở
Tuần 4-5 (xem chunker.py, run_dev_qa.py, báo cáo Tuần 4 mục 3.4) về 1 nơi
duy nhất, để mọi script (run_dev_qa.py, test_longcontext_baseline.py, script
Tuần 6-8 sau này) đọc chung, tránh lệch tham số giữa các lần chạy.

Nếu sau này đổi 1 giá trị nào ở đây, PHẢI ghi lại lý do đổi (giống style
comment trong chunker.py: "MAX_TOKENS_PER_CHUNK tăng 256 -> 320 (Tuần 5, sau
oracle-context test Tuần 4)...") — không đổi âm thầm.
"""

from __future__ import annotations

# ----- Model sinh câu trả lời (generation) -----
# Khóa chặt 1 model duy nhất cho toàn bộ đánh giá cuối (không cascade sang
# model khác), để tránh confound đã ghi nhận ở Tuần 4 (model cascade
# gemini-3.6-flash/3.5-flash/3.5-flash-lite tạo nhiễu giữa các cấu hình).
MODEL_NAME = "gemini-3.5-flash-lite"

# ----- Ngưỡng abstention tầng Retrieval (tau) -----
# = Kịch bản 1 "Two-Tier Abstention" trong run_dev_qa.py (cờ --tau 0.38),
# KHÔNG phải Kịch bản 2 "Strict Sweep Filter" (page_aware=0.5/fixed_size=0.45)
# vốn là default hiện tại của script. Chọn Kịch bản 1 làm giá trị CHÍNH THỨC
# vì đây là ngưỡng DÙNG CHUNG cho mọi cấu hình (page_aware, fixed_size, và
# cả long-context nếu áp dụng tau ở tầng model refusal) — đúng yêu cầu đề
# cương "chênh lệch kết quả chỉ phản ánh đúng biến ranh giới chunk", không
# lẫn thêm biến "ngưỡng khác nhau giữa 2 cấu hình".
# Nguồn: suy ra từ khoảng trống giữa cosine thấp nhất của câu answerable
# (dev_13=0.3994) và cosine cao nhất của 1 câu unanswerable cụ thể
# (dev_32=0.3723) trong dev_retrieval_raw.csv.
TAU = 0.38

# ----- Top-k đoạn đưa vào tầng generation (KHÔNG phải Hit@k ở tầng retrieval) -----
# Hit@3 ở run_dev_retrieval.py vẫn giữ k=3 làm chỉ số so sánh chunking gốc,
# KHÔNG đổi theo giá trị này. k=15 chỉ áp dụng khi đưa đoạn vào Gemini để
# sinh câu trả lời (xem comment chi tiết trong run_dev_qa.py về việc đã thử
# k=5 nhưng không hiệu quả, k=15 kết hợp chunk=320 mới sửa được dev_10/19/25).
TOP_K_GENERATION = 15

# ----- Chunking -----
CHUNK_MAX_TOKENS = 320          # chunker.py: MAX_TOKENS_PER_CHUNK
CHUNK_OVERLAP_WORDS = 30        # chunker.py: OVERLAP_WORDS
BRIDGE_WORDS_EACH_SIDE = 128    # chunker.py: BRIDGE_WORDS_EACH_SIDE
FIXED_CHUNK_WORDS = 170         # chunker.py: FIXED_CHUNK_WORDS (baseline fixed_size)

# Kiến trúc mặc định đã duyệt của ứng dụng (dùng khi ứng dụng thật chỉ chạy
# 1 chiến lược, không phải khi so sánh 3 cấu hình trong lúc đánh giá).
DEFAULT_CHUNKING_STRATEGY = "page_aware"

# ----- Model embedding (tầng retrieval) -----
EMBED_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# ----- Sinh câu trả lời -----
GENERATION_TEMPERATURE = 0.0
MODEL_ABSTAIN_TEXT = "Không tìm thấy thông tin trong tài liệu."

# ----- Bộ câu hỏi -----
DEV_SET_PATH = "data/eval_sets/dev_questions_normalized.json"
CORPUS_DIR = "data/corpus"


if __name__ == "__main__":
    # In lại toàn bộ giá trị đã khóa, để đối chiếu nhanh trước khi chạy A2-A4.
    for name, value in list(globals().items()):
        if name.isupper():
            print(f"{name} = {value!r}")
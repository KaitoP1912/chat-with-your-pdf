"""
demo_retrieval.py — Bước 8: demo end-to-end, dùng để nghiệm thu Tuần 3

Luồng: 1 file PDF -> build_clean_pages -> chunk_by_page -> embed + FAISS -> search
       -> in ra top-k chunk kèm số trang.

Cách chạy (từ thư mục gốc project, đã activate venv):
    python script/pilot_tuan3/demo_retrieval.py "data/corpus/normal_hienphap.pdf" "Vai trò của Mặt trận Tổ quốc Việt Nam là gì?"

Yêu cầu: đã cài đủ py_vncorenlp (+ model), sentence-transformers, faiss-cpu, transformers.
Lần chạy đầu tiên cần mạng để tự tải model vietnamese-bi-encoder + tokenizer.
"""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

# Cho phép chạy trực tiếp file này từ bất kỳ đâu: tự thêm thư mục gốc project
# (2 cấp trên file này: script/pilot_tuan3/demo_retrieval.py -> <root>) vào
# sys.path, để Python tìm thấy package "source".
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ĐƯỜNG DẪN TUYỆT ĐỐI tới thư mục chứa model VnCoreNLP (đã tải ở Tuần 2).
# QUAN TRỌNG: py_vncorenlp cần đường dẫn TUYỆT ĐỐI, đường dẫn tương đối gây lỗi
# JVM "NoClassDefFoundError: vn/pipeline/VnCoreNLP" (đã gặp thực tế).
# Nếu thư mục model của bạn nằm ở vị trí khác, SỬA DÒNG DƯỚI cho đúng.
VNCORENLP_DIR = str(PROJECT_ROOT / "vncorenlp_models")

RESULTS_DIR = PROJECT_ROOT / "results" / "tuan3_pilot"

sys.path.insert(0, str(PROJECT_ROOT))

from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.chunker import chunk_by_page, default_token_counter
from source.retrieval.vectorstore import build_index, search


class _Tee:
    """In ra console NHƯ BÌNH THƯỜNG, đồng thời ghi lại vào buffer để lưu file.
    Không đổi cách hiển thị khi chạy, chỉ thêm khả năng lưu log.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def main() -> None:
    if len(sys.argv) < 3:
        print('Cách dùng: python script/pilot_tuan3/demo_retrieval.py <file.pdf> "<câu hỏi>"')
        sys.exit(1)

    file_path = sys.argv[1]
    question = sys.argv[2]

    print(f"[1/4] Đọc + chuẩn hóa PDF: {file_path}")
    t0 = time.time()
    pages = build_clean_pages(file_path)
    print(f"      -> {len(pages)} trang, {time.time()-t0:.2f}s")

    print("[2/4] Chunking (page-aware + bridge chunk)")
    t0 = time.time()
    token_counter = default_token_counter()  # cần mạng lần đầu (tải tokenizer)
    chunks = chunk_by_page(pages, token_counter=token_counter)
    n_bridge = sum(1 for c in chunks if c["is_bridge"])
    print(f"      -> {len(chunks)} chunk (gồm {n_bridge} bridge chunk), {time.time()-t0:.2f}s")

    print("[3/4] Embedding + lập chỉ mục FAISS")
    t0 = time.time()
    index = build_index(chunks, VNCORENLP_DIR)  # cần mạng lần đầu (tải model embedding)
    print(f"      -> Index có {index._index.ntotal} vector, {time.time()-t0:.2f}s")

    print(f"[4/4] Tìm kiếm cho câu hỏi: {question!r}")
    t0 = time.time()
    hits = search(index, question, VNCORENLP_DIR, k=3)
    print(f"      -> {time.time()-t0:.2f}s\n")

    print("=== KẾT QUẢ TOP-3 ===")
    for rank, hit in enumerate(hits, start=1):
        page_info = f"trang {hit.page_number}" if not hit.is_bridge else f"trang {hit.page_range} (bridge)"
        print(f"\n#{rank} | {page_info} | score={hit.score:.4f} | chunk_id={hit.chunk_id}")
        preview = hit.text[:200].replace("\n", " ")
        print(f"    {preview}...")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"demo_retrieval_{timestamp}.txt"

    log_buffer = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, log_buffer)
    try:
        main()
    finally:
        sys.stdout = real_stdout
        log_path.write_text(log_buffer.getvalue(), encoding="utf-8")
        print(f"\n[Đã lưu log tại: {log_path}]")
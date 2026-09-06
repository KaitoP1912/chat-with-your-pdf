"""
script/app_cli.py — Tuần 5, Việc 2: Kết nối luồng End-to-End tối thiểu

Chạy toàn bộ pipeline "Chat with Your PDF" từ dòng lệnh:
  đọc PDF -> chuẩn hóa -> chia đoạn (chunk_by_page) -> embedding + index
  (Hybrid FAISS + BM25) -> tìm kiếm -> gọi Gemini -> in câu trả lời kèm
  trích dẫn trang.

CHỈ dùng lại các hàm có sẵn của source/ (build_clean_pages, chunk_by_page,
build_index, search, generate_answer) — không viết lại logic bên trong.

Tham số ĐÃ KHÓA ở Tuần 4, dùng làm mặc định (có thể override để debug, nhưng
KHÔNG override khi demo nghiệm thu):
  - tau   = 0.38  (Kịch bản B)
  - k     = 15
  - chunking = chunk_by_page (kiến trúc chính thức, KHÔNG dùng chunk_fixed_size)

Cách dùng:
  python script/app_cli.py --pdf data/corpus/normal_hienphap_33tr.pdf \
      --question "Nhiệm kỳ Quốc hội là bao nhiêu năm?"

Tham số tùy chọn:
  --vncorenlp_dir   Thư mục model VnCoreNLP (mặc định: ./vncorenlp_models)
  --corpus_dir      Thư mục chứa PDF gốc, dùng để render ảnh trang biểu đồ
                     nếu câu trả lời rơi vào CHART_HEAVY_PAGES_MANUAL
                     (mặc định: thư mục chứa chính file --pdf)
  --k               Số đoạn lấy về cho QA (mặc định 15, đã khóa)
  --tau             Ngưỡng Dense Guardrail (mặc định 0.38, đã khóa)
  --model           Model Gemini (mặc định gemini-3.5-flash-lite, đã khóa)

LƯU Ý: script này KHÔNG xử lý lỗi phức tạp, KHÔNG tối ưu tốc độ — mục đích
duy nhất là DEMO chứng minh hệ thống chạy được từ đầu đến cuối (nghiệm thu
Tuần 5). Giao diện MVP thật là việc của Tuần 6.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Cho phép chạy trực tiếp "python script/app_cli.py ..." từ thư mục gốc repo
# mà không cần cài package / set PYTHONPATH thủ công.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.ingestion.pdf_loader import load_pdf_pages
from source.ingestion.scan_detector import detect_scan, should_reject
from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.chunker import chunk_by_page
from source.retrieval.vectorstore import build_index, search
from source.qa.qa_generator import generate_answer, DEFAULT_MODEL

# --- Tham số đã khóa Tuần 4 (KHÔNG đổi khi demo chính thức) ---
LOCKED_TAU = 0.38
LOCKED_K = 15
LOCKED_MODEL = DEFAULT_MODEL  # "gemini-3.5-flash-lite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with Your PDF — chạy end-to-end 1 câu hỏi trên 1 file PDF."
    )
    parser.add_argument("--pdf", required=True, help="Đường dẫn file PDF cần hỏi đáp.")
    parser.add_argument("--question", required=True, help="Câu hỏi tiếng Việt.")
    parser.add_argument(
        "--vncorenlp_dir",
        default="./vncorenlp_models",
        help="Thư mục model VnCoreNLP (mặc định: ./vncorenlp_models).",
    )
    parser.add_argument(
        "--corpus_dir",
        default=None,
        help="Thư mục chứa PDF gốc (để render ảnh trang biểu đồ). "
        "Mặc định: thư mục chứa chính file --pdf.",
    )
    parser.add_argument("--k", type=int, default=LOCKED_K, help=f"Số đoạn lấy về (mặc định {LOCKED_K}, đã khóa).")
    parser.add_argument("--tau", type=float, default=LOCKED_TAU, help=f"Ngưỡng tau (mặc định {LOCKED_TAU}, đã khóa).")
    parser.add_argument("--model", default=LOCKED_MODEL, help=f"Model Gemini (mặc định {LOCKED_MODEL}).")
    return parser.parse_args()


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    args = parse_args()

    pdf_path = args.pdf
    corpus_dir = args.corpus_dir or os.path.dirname(os.path.abspath(pdf_path)) or "."

    _print_header("BƯỚC 1/4 — Đọc & chuẩn hóa PDF (Trạm 1 + ingest_glue)")
    t0 = time.time()
    try:
        raw_pages = load_pdf_pages(pdf_path)
        scan_result = detect_scan(raw_pages)

        if should_reject(scan_result):
            print(
                f"[LỖI] File '{pdf_path}' là PDF scan hoàn toàn "
                "(không có lớp text) — ứng dụng hiện chỉ hỗ trợ PDF có text layer. "
                "Vui lòng dùng OCR trước khi tải lên."
            )
            sys.exit(1)

        pages = build_clean_pages(pdf_path)
    except Exception as exc:
        print(f"[LỖI] Không đọc được file PDF '{pdf_path}': {exc}")
        sys.exit(1)
    print(f"Đã đọc {len(pages)} trang từ '{os.path.basename(pdf_path)}' "
          f"({time.time() - t0:.2f}s)")

    _print_header("BƯỚC 2/4 — Chia đoạn (chunk_by_page, chunk=320, kiến trúc chính thức)")
    t0 = time.time()
    chunks = chunk_by_page(pages)
    n_bridge = sum(1 for c in chunks if c["is_bridge"])
    print(f"Đã tạo {len(chunks)} chunk (trong đó {n_bridge} bridge chunk) "
          f"({time.time() - t0:.2f}s)")

    _print_header("BƯỚC 3/4 — Embedding + dựng Index (Hybrid FAISS + BM25)")
    t0 = time.time()
    index = build_index(chunks, args.vncorenlp_dir)
    print(f"Đã dựng index cho {len(chunks)} chunk ({time.time() - t0:.2f}s)")

    _print_header(f"BƯỚC 4/4 — Tìm kiếm (k={args.k}) + Gọi Gemini (tau={args.tau}, model={args.model})")
    t0 = time.time()
    hits = search(index, args.question, args.vncorenlp_dir, k=args.k)
    answer = generate_answer(
        args.question,
        hits,
        tau=args.tau,
        target_model=args.model,
        corpus_dir=corpus_dir,
    )
    print(f"Đã có câu trả lời ({time.time() - t0:.2f}s)")

    # --- In kết quả cuối cùng, rõ ràng, dễ đọc ---
    _print_header("KẾT QUẢ")
    print(f"Câu hỏi : {args.question}")
    print(f"File     : {os.path.basename(pdf_path)}")
    print("-" * 70)

    if answer.is_error:
        print(f"[LỖI GỌI GEMINI] {answer.error_message}")
        sys.exit(1)

    print(f"Trả lời  : {answer.answer_text}")
    print(f"Từ chối  : {'CÓ' if answer.is_abstained else 'KHÔNG'}"
          + (f" (lý do: {answer.abstain_reason})" if answer.is_abstained else ""))

    if answer.is_abstained:
        print("Trích dẫn: (không có, do hệ thống từ chối trả lời)")
    else:
        if answer.citations:
            pages_str = []
            for c in answer.citations:
                if c["page_number"] is not None:
                    pages_str.append(f"trang {c['page_number']}" + (" [bridge nội bộ]" if c["is_bridge"] else ""))
                elif c["page_range"]:
                    pages_str.append(f"trang {c['page_range']} [bridge liên trang]")
            print(f"Trích dẫn: {', '.join(pages_str) if pages_str else '(không xác định được trang)'}")
        else:
            print("Trích dẫn: (không có)")

    if answer.chart_pages_sent:
        print(f"Ảnh trang đã gửi kèm Gemini: {', '.join(answer.chart_pages_sent)}")

    print(f"Model    : {answer.model_used}")
    if answer.latency_seconds is not None:
        print(f"Độ trễ Gemini: {answer.latency_seconds}s")
    if answer.total_tokens is not None:
        print(f"Token    : prompt={answer.prompt_tokens}, output={answer.output_tokens}, tổng={answer.total_tokens}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
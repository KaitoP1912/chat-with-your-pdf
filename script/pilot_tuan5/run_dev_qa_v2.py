"""
script/pilot_tuan5/run_dev_qa_v2.py — Tuần 5, kiểm tra 2 thay đổi trong
source/qa/qa_generator.py:

  Việc 1: Tích hợp gửi ảnh trang biểu đồ (CHART_HEAVY_PAGES_MANUAL) thẳng vào
          generate_answer() của pipeline chính thức.
  Việc 2: Nới QUY TẮC 2 trong build_qa_prompt() để giảm Model Over-Refusal,
          cho phép trả lời kèm ghi chú không chắc chắn khi có ít nhất 1 đoạn
          liên quan, thay vì từ chối thẳng.

Bản sao có điều chỉnh của script/pilot_tuan4/run_dev_qa.py (CHỈ ĐỌC file gốc
để tham khảo, không sửa file đó). Khác biệt so với bản gốc:
  - Chỉ chạy đúng 1 cấu hình: page_aware (đã khóa làm kiến trúc chính thức,
    không cần chạy lại fixed_size ở Tuần 5).
  - Tham số đã khóa cố định: --tau 0.38 --k 15 (không hỗ trợ 2 kịch bản tau
    song song như bản Tuần 4 nữa — Tuần 5 chỉ còn 1 cấu hình production).
  - Truyền corpus_dir vào generate_answer() để qa_generator.py render được
    ảnh trang biểu đồ khi cần (Việc 1).
  - Thêm cột chart_pages_sent vào CSV để kiểm chứng dev_05 có thật sự nhận
    được ảnh trang 7 hay không.
  - In thêm thống kê so sánh tỉ lệ model_refusal và false acceptance so với
    kết quả Tuần 4 (dev_qa_results_FINAL_v3.csv), phục vụ tóm tắt yêu cầu.

QUAN TRỌNG: theo đúng cảnh báo trong nhiệm vụ — nới lỏng prompt có rủi ro làm
TĂNG false acceptance (câu unanswerable bị trả lời liều). Tuần 4 đạt 0/11,
đây là thành tích quan trọng nhất, KHÔNG được làm tệ đi. Script này tự động
in cảnh báo rõ ràng nếu phát hiện false acceptance > 0 sau khi chạy.

Cách chạy (dry-run 5 câu trước, đúng quy trình đã thống nhất — luôn kiểm tra
bước rẻ trước khi tốn Gemini API):
    python script/pilot_tuan5/run_dev_qa_v2.py --limit 5

Sau khi xác nhận ổn, chạy full 34 câu:
    python script/pilot_tuan5/run_dev_qa_v2.py --limit 0 \
        --out results/tuan5_pilot/dev_qa_results_v2_page_aware.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# run_dev_retrieval.py nằm ở script/pilot_tuan4/ — CHỈ ĐỌC để tái dùng CONFIGS,
# build_indices_for_file, is_hit; KHÔNG sửa file đó (người khác đang dùng song song).
sys.path.insert(0, str(PROJECT_ROOT / "script" / "pilot_tuan4"))

from source.retrieval.vectorstore import search, ChunkIndex
from source.qa.qa_generator import generate_answer, DEFAULT_MODEL
from run_dev_retrieval import build_indices_for_file, is_hit

# Tuần 5: chỉ còn 1 cấu hình chính thức, không chạy song song fixed_size nữa.
CONFIG_NAME = "page_aware"

# Tham số đã khóa từ Tuần 4, không tự đổi khi chạy script này.
TAU_LOCKED = 0.38
K_LOCKED = 15


def main():
    parser = argparse.ArgumentParser(
        description="Tuần 5: chạy lại QA (page_aware) với qa_generator.py đã "
                     "tích hợp ảnh trang biểu đồ + prompt nới over-refusal."
    )
    parser.add_argument("--dev-set", default="data/eval_sets/dev_questions_normalized.json")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--vncorenlp-dir", default="vncorenlp_models")
    parser.add_argument("--k", type=int, default=K_LOCKED,
                         help=f"Mặc định {K_LOCKED} (đã khóa Tuần 4, không nên đổi).")
    parser.add_argument("--tau", type=float, default=TAU_LOCKED,
                         help=f"Mặc định {TAU_LOCKED} (Kịch bản B, đã khóa Tuần 4, không nên đổi).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="results/tuan5_pilot/dev_qa_results_v2_page_aware.csv")
    parser.add_argument("--limit", type=int, default=5,
                         help="Chỉ chạy N câu đầu để dry-run (mặc định 5). "
                              "Dùng --limit 0 để chạy FULL 34 câu.")
    parser.add_argument("--sleep", type=float, default=5.0,
                         help="Số giây nghỉ giữa 2 lần gọi Gemini (15 RPM cho flash-lite).")
    args = parser.parse_args()

    print(f"=== TUẦN 5 — QA v2 (page_aware, tau={args.tau}, k={args.k}) ===")
    print(f"Output: {args.out}\n")

    corpus_dir = Path(args.corpus_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vncorenlp_dir = str(Path(args.vncorenlp_dir).resolve())

    with open(Path(args.dev_set).resolve(), "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    questions = dev_data["questions"]

    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
        print(f"*** DRY-RUN: chỉ chạy {len(questions)} câu đầu tiên. "
              f"Dùng --limit 0 để chạy full sau khi kiểm tra ổn. ***\n")

    unique_files = sorted({q["source_file"] for q in questions})
    print(f"Build index cho {len(unique_files)} file: {unique_files}\n")

    all_indices: Dict[str, Dict[str, ChunkIndex]] = {}
    for source_file in unique_files:
        print(f"[Build index] {source_file}")
        all_indices[source_file] = build_indices_for_file(source_file, corpus_dir, vncorenlp_dir)
        print()

    rows = []
    total = len(questions)

    for done, q in enumerate(questions, start=1):
        source_file = q["source_file"]
        expected_pages = [int(p) for p in (q.get("expected_page") or [])] if q.get("expected_page") else []
        index = all_indices[source_file][CONFIG_NAME]

        hits = search(index, q["question"], vncorenlp_dir, k=args.k)
        answer = generate_answer(
            q["question"], hits, tau=args.tau, target_model=args.model,
            corpus_dir=str(corpus_dir),
        )

        citation_correct = ""
        if q["is_answerable"] and not answer.is_abstained and not answer.is_error:
            any_correct = any(
                is_hit(c.get("page_number"), c.get("page_range"), expected_pages)
                for c in answer.citations
            )
            citation_correct = any_correct

        rows.append({
            "id": q["id"],
            "config": CONFIG_NAME,
            "tau_used": args.tau,
            "is_answerable": q["is_answerable"],
            "type": q.get("type", ""),
            "question": q["question"],
            "expected_page": ";".join(str(p) for p in expected_pages),
            "is_abstained": answer.is_abstained,
            "abstain_reason": answer.abstain_reason or "",
            "is_error": answer.is_error,
            "error_message": answer.error_message or "",
            "answer_text": answer.answer_text,
            "citations": json.dumps(answer.citations, ensure_ascii=False),
            "citation_correct": citation_correct,
            "chart_pages_sent": ";".join(answer.chart_pages_sent),
            "latency_seconds": answer.latency_seconds if answer.latency_seconds is not None else "",
            "prompt_tokens": answer.prompt_tokens if answer.prompt_tokens is not None else "",
            "output_tokens": answer.output_tokens if answer.output_tokens is not None else "",
            "total_tokens": answer.total_tokens if answer.total_tokens is not None else "",
            "model_used": answer.model_used or "",
            "answer_correctness_manual": "",
        })

        status = "ABSTAIN" if answer.is_abstained else ("ERROR" if answer.is_error else "OK")
        chart_note = f" [ảnh: {answer.chart_pages_sent}]" if answer.chart_pages_sent else ""
        print(f"  [{done}/{total}] {q['id']:8s} -> {status} ({answer.abstain_reason or '-'})"
              f" latency={answer.latency_seconds} tokens={answer.total_tokens}{chart_note}")

        if answer.abstain_reason != "retrieval_threshold":
            time.sleep(args.sleep)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nĐã ghi {len(rows)} dòng vào: {out_path}")

    # --- Thống kê phục vụ tóm tắt yêu cầu: model_refusal + false acceptance ---
    answerable_rows = [r for r in rows if str(r["is_answerable"]).strip().lower() == "true"]
    n_model_refusal = sum(1 for r in answerable_rows if r["abstain_reason"] == "model_refusal")
    print(f"\n[THỐNG KÊ] model_refusal trên câu answerable: {n_model_refusal}/{len(answerable_rows)}")

    unans_rows = [r for r in rows if str(r["is_answerable"]).strip().lower() == "false"]
    false_acceptance = [
        r for r in unans_rows
        if not (str(r["is_abstained"]).strip().lower() == "true")
    ]
    if false_acceptance:
        print(f"\n[CẢNH BÁO NGHIÊM TRỌNG] {len(false_acceptance)}/{len(unans_rows)} câu unanswerable "
              f"KHÔNG bị chặn (false acceptance) — Tuần 4 đang là 0/11, đây LÀ THOÁI LUI, "
              f"PHẢI báo cáo rõ trong tóm tắt, KHÔNG được che giấu:")
        for r in false_acceptance:
            print(f"   - {r['id']}: answer_text={r['answer_text'][:80]!r}")
    else:
        print(f"[OK] False acceptance vẫn là 0/{len(unans_rows)}, không có thoái lui so với Tuần 4.")

    dev05_rows = [r for r in rows if r["id"] == "dev_05"]
    if dev05_rows:
        r = dev05_rows[0]
        print(f"\n[dev_05] is_abstained={r['is_abstained']} abstain_reason={r['abstain_reason']!r} "
              f"chart_pages_sent={r['chart_pages_sent']!r}")

    print("\nBước tiếp theo: mở CSV, điền cột answer_correctness_manual cho các dòng có answer_text, "
          "rồi so sánh với results/tuan4_pilot/dev_qa_results_FINAL_v3.csv để viết tóm tắt Tuần 5.")


if __name__ == "__main__":
    main()
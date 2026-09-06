"""
script/pilot_tuan6/run_test_qa.py — Tuần 6: Đánh giá CHÍNH THỨC trên Test Set
(25 câu, ĐỘC LẬP với Dev Set), cho 2 cấu hình RAG: page_aware và fixed_size.

Baseline long-context (baseline B) đã có sẵn ở
script/pilot_tuan5/run_longcontext_baseline.py — dùng lại nguyên script đó,
chỉ đổi --dev-set thành đường dẫn tới test_questions.json, ví dụ:

    python script/pilot_tuan5/run_longcontext_baseline.py \
        --dev-set data/eval_sets/test_questions.json --limit 0 \
        --out results/tuan6_pilot/test_qa_results_longcontext.csv

Script NÀY (run_test_qa.py) chạy 2 cấu hình còn lại (page_aware, fixed_size).

Cách hoạt động:
  1. Với mỗi file nguồn DUY NHẤT xuất hiện trong test set: build_clean_pages()
     1 lần, chunk 1 lần (theo --strategy), build_index() 1 lần — không build
     lại cho từng câu hỏi (đúng tinh thần tối ưu của run_longcontext_baseline.py).
  2. Với mỗi câu hỏi:
       - Hit@3: search hybrid với k=3 CỐ ĐỊNH (đúng chỉ số Hit@3 đã dùng
         xuyên suốt từ Tuần 4 để so sánh 2 chiến lược chunking — KHÔNG đổi
         theo TOP_K_GENERATION=15 của config.py, hai việc này đo hai thứ
         khác nhau: Hit@3 đo retriever, k=15 chỉ dùng để đưa ngữ cảnh cho
         Gemini sinh câu trả lời).
       - Generation: search hybrid k=15 (config.TOP_K_GENERATION, đã khóa)
         rồi generate_answer() với tau=0.38 (config.TAU, đã khóa).
  3. Ghi CSV CÙNG SCHEMA với dev_qa_results_longcontext.csv (chỉ thêm cột
     hit_at_3) để gộp cả 3 cấu hình thành 1 bảng so sánh duy nhất.

QUAN TRỌNG: KHÔNG dùng script này để tinh chỉnh tau/k sau khi thấy kết quả.
Các giá trị này đã khóa ở config.py từ Tuần 4-5 (xem comment chi tiết trong
config.py). Nếu kết quả tệ ở 1 câu nào đó, đó LÀ một kết quả cần đưa vào
error analysis, không phải lý do để đổi tham số rồi chạy lại.

Cách dùng:
    # Dry-run 3 câu đầu để kiểm tra script chạy được trước khi tốn quota:
    python script/pilot_tuan6/run_test_qa.py --strategy page_aware --limit 3

    # Chạy full 25 câu, 1 cấu hình mỗi lần gọi:
    python script/pilot_tuan6/run_test_qa.py --strategy page_aware --limit 0 \
        --out results/tuan6_pilot/test_qa_results_page_aware.csv

    python script/pilot_tuan6/run_test_qa.py --strategy fixed_size --limit 0 \
        --out results/tuan6_pilot/test_qa_results_fixed_size.csv

    # Nếu bị đứt giữa chừng (rate limit, mất mạng...), chạy lại với --resume
    # để bỏ qua các câu đã thành công:
    python script/pilot_tuan6/run_test_qa.py --strategy page_aware --limit 0 \
        --out results/tuan6_pilot/test_qa_results_page_aware.csv --resume
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

import config  # noqa: E402  (đọc tham số đã khóa: TAU, TOP_K_GENERATION, MODEL_NAME, CORPUS_DIR)

from source.retrieval.ingest_glue import build_clean_pages  # noqa: E402
from source.retrieval.chunker import chunk_by_page, chunk_fixed_size  # noqa: E402
from source.retrieval.vectorstore import build_index, search  # noqa: E402
from source.qa.qa_generator import generate_answer  # noqa: E402

STRATEGIES = {
    "page_aware": chunk_by_page,
    "fixed_size": chunk_fixed_size,
}

# Cố định Hit@3 xuyên suốt để so sánh đúng biến "ranh giới chunk" (đúng comment
# trong config.py: "Hit@3 ở run_dev_retrieval.py vẫn giữ k=3 làm chỉ số so
# sánh chunking gốc, KHÔNG đổi theo giá trị [TOP_K_GENERATION]").
HIT_AT_K = 3


def hit_at_k(hits, expected_pages: List[int], k: int) -> bool:
    """True nếu ít nhất 1 trong k chunk đầu bao phủ ít nhất 1 trang expected."""
    if not expected_pages:
        return False
    covered = set()
    for h in hits[:k]:
        if h.page_number is not None:
            covered.add(h.page_number)
        elif h.page_range:
            try:
                a, b = h.page_range.split("-")
                covered.add(int(a))
                covered.add(int(b))
            except ValueError:
                pass
    return bool(covered & set(expected_pages))


def extract_predicted_pages(citations: List[dict]) -> List[int]:
    pages: List[int] = []
    for c in citations:
        if c.get("page_number") is not None:
            pages.append(int(c["page_number"]))
        elif c.get("page_range"):
            try:
                a, b = c["page_range"].split("-")
                pages.extend([int(a), int(b)])
            except ValueError:
                pass
    return sorted(set(pages))


def empty_row(q: dict, config_label: str, tau: float, model: str, error_message: str) -> dict:
    expected_pages = [int(p) for p in (q.get("expected_page") or [])]
    return {
        "id": q["id"], "config": config_label, "tau_used": tau,
        "is_answerable": q["is_answerable"], "type": q.get("type", ""),
        "question": q["question"],
        "expected_page": ";".join(str(p) for p in expected_pages),
        "hit_at_3": "", "is_abstained": "", "abstain_reason": "",
        "is_error": True, "error_message": error_message,
        "answer_text": "", "citations": "[]", "citation_correct": "",
        "latency_seconds": "", "prompt_tokens": "", "output_tokens": "",
        "total_tokens": "", "model_used": model,
        "answer_correctness_manual": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá chính thức Test Set — cấu hình page_aware hoặc fixed_size."
    )
    parser.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    parser.add_argument("--test-set", default="data/eval_sets/test_questions.json")
    parser.add_argument("--corpus-dir", default=config.CORPUS_DIR)
    parser.add_argument("--vncorenlp_dir", default="./vncorenlp_models")
    parser.add_argument("--k", type=int, default=config.TOP_K_GENERATION,
                         help=f"Top-k đưa vào generation (mặc định {config.TOP_K_GENERATION}, đã khóa).")
    parser.add_argument("--tau", type=float, default=config.TAU,
                         help=f"Ngưỡng abstention (mặc định {config.TAU}, đã khóa).")
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=3,
                         help="Chỉ chạy N câu đầu (mặc định 3, dry-run). --limit 0 = chạy full.")
    parser.add_argument("--sleep", type=float, default=5.0,
                         help="Số giây nghỉ giữa 2 lần gọi Gemini (model flash-lite 15 RPM).")
    parser.add_argument("--resume", action="store_true",
                         help="Bỏ qua các câu đã thành công ở lần chạy trước (đọc lại --out cũ).")
    args = parser.parse_args()

    chunk_fn = STRATEGIES[args.strategy]
    out_path = Path(args.out or f"results/tuan6_pilot/test_qa_results_{args.strategy}.csv").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(Path(args.test_set).resolve(), "r", encoding="utf-8") as f:
        test_data = json.load(f)
    questions = test_data["questions"]

    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
        print(f"*** DRY-RUN: chỉ chạy {len(questions)} câu đầu. Dùng --limit 0 để chạy full. ***\n")

    previous_ok: Dict[str, dict] = {}
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        for r in old_rows:
            if str(r.get("is_error", "")).strip().lower() != "true":
                previous_ok[r["id"]] = r
        print(f"*** RESUME: {len(previous_ok)} dòng đã có sẵn, sẽ bỏ qua. ***\n")

    corpus_dir = Path(args.corpus_dir).resolve()
    unique_files = sorted({q["source_file"] for q in questions})
    print(f"=== TEST SET — strategy={args.strategy}, k={args.k}, tau={args.tau}, model={args.model} ===")
    print(f"Dựng index cho {len(unique_files)} file nguồn: {unique_files}\n")

    indexes: Dict[str, object] = {}
    build_errors: Dict[str, str] = {}
    for source_file in unique_files:
        pdf_path = corpus_dir / source_file
        print(f"[Dựng index] {source_file} ({args.strategy}) ...")
        try:
            pages = build_clean_pages(str(pdf_path))
            chunks = chunk_fn(pages)
            index = build_index(chunks, args.vncorenlp_dir)
            indexes[source_file] = index
            n_bridge = sum(1 for c in chunks if c.get("is_bridge"))
            print(f"  -> {len(pages)} trang, {len(chunks)} chunk ({n_bridge} bridge)\n")
        except Exception as e:
            build_errors[source_file] = str(e)
            print(f"  -> LỖI dựng index (ghi nhận, không dừng script): {e}\n")

    rows: List[dict] = []
    total = len(questions)
    done = 0

    for q in questions:
        done += 1
        source_file = q["source_file"]
        expected_pages = [int(p) for p in (q.get("expected_page") or [])]

        if q["id"] in previous_ok:
            rows.append(previous_ok[q["id"]])
            print(f"  [{done}/{total}] {q['id']:8s} -> REUSED")
            continue

        if source_file in build_errors:
            rows.append(empty_row(
                q, args.strategy, args.tau, args.model,
                f"loi_dung_index: {build_errors[source_file]}",
            ))
            print(f"  [{done}/{total}] {q['id']:8s} -> ERROR (build index)")
            continue

        index = indexes[source_file]
        try:
            # Hit@3: đo retriever riêng biệt với k=3 cố định.
            hits_at_3 = search(index, q["question"], args.vncorenlp_dir, k=HIT_AT_K)
            hit3 = hit_at_k(hits_at_3, expected_pages, HIT_AT_K)

            # Generation: k=15 (đã khóa) -> generate_answer với tau đã khóa.
            hits_for_gen = search(index, q["question"], args.vncorenlp_dir, k=args.k)
            answer = generate_answer(
                q["question"], hits_for_gen, tau=args.tau,
                target_model=args.model, corpus_dir=str(corpus_dir),
            )

            predicted_pages = (
                extract_predicted_pages(answer.citations) if not answer.is_abstained else []
            )

            citation_correct = ""
            if q["is_answerable"] and not answer.is_abstained and not answer.is_error:
                citation_correct = bool(set(expected_pages) & set(predicted_pages))

            rows.append({
                "id": q["id"], "config": args.strategy, "tau_used": args.tau,
                "is_answerable": q["is_answerable"], "type": q.get("type", ""),
                "question": q["question"],
                "expected_page": ";".join(str(p) for p in expected_pages),
                "hit_at_3": hit3,
                "is_abstained": answer.is_abstained,
                "abstain_reason": answer.abstain_reason or "",
                "is_error": answer.is_error,
                "error_message": answer.error_message or "",
                "answer_text": answer.answer_text,
                "citations": json.dumps(answer.citations, ensure_ascii=False),
                "citation_correct": citation_correct,
                "latency_seconds": answer.latency_seconds if answer.latency_seconds is not None else "",
                "prompt_tokens": answer.prompt_tokens if answer.prompt_tokens is not None else "",
                "output_tokens": answer.output_tokens if answer.output_tokens is not None else "",
                "total_tokens": answer.total_tokens if answer.total_tokens is not None else "",
                "model_used": answer.model_used or args.model,
                "answer_correctness_manual": "",
            })

            status = "ABSTAIN" if answer.is_abstained else ("ERROR" if answer.is_error else "OK")
            print(f"  [{done}/{total}] {q['id']:8s} -> {status:8s} hit@3={hit3} "
                  f"latency={answer.latency_seconds} tokens={answer.total_tokens}")

            if not answer.is_error:
                time.sleep(args.sleep)

        except Exception as e:
            rows.append(empty_row(q, args.strategy, args.tau, args.model, f"loi_khong_luong_truoc: {e}"))
            print(f"  [{done}/{total}] {q['id']:8s} -> LỖI KHÔNG LƯỜNG TRƯỚC: {e}")

    if not rows:
        print("Không có dòng nào để ghi (0 câu hỏi).")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(1 for r in rows if str(r["is_error"]).strip().lower() == "true")
    n_hit3 = sum(1 for r in rows if str(r.get("hit_at_3")).strip().lower() == "true")
    print(f"\nĐã ghi {len(rows)} dòng vào: {out_path}")
    print(f"Hit@3: {n_hit3}/{len(rows)} | Lỗi: {n_errors}/{len(rows)}")
    print("\nBước tiếp theo: điền cột answer_correctness_manual (đúng hoàn toàn / đúng một phần / sai) "
          "cho các dòng có answer_text, rồi chạy aggregate_results.py để tổng hợp bảng so sánh 3 cấu hình.")


if __name__ == "__main__":
    main()
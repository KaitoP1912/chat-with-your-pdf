"""
script/tuan_bo_sung/run_ablation_fair_baseline.py — Ablation #2 (Nhóm A):
baseline fixed-size công bằng (fixed_size_fair) và page-aware không bridge
(page_aware_no_bridge).

KHÔNG ghi đè bất kỳ file đã khóa nào (results/tuan6_pilot/,
data/eval_sets/test_questions.json). Chỉ đọc test set, ghi kết quả MỚI vào
results/tuan_bo_sung/ablation_fair_baseline/.

Tái dùng nguyên logic chấm điểm (hit_at_k, extract_predicted_pages,
empty_row, HIT_AT_K) từ script/pilot_tuan6/run_test_qa.py bằng import —
không viết lại để tránh lệch logic chấm so với Test Set gốc.

Chỉ chạy trên các câu is_answerable=True của Test Set (citation_correct
chỉ có ý nghĩa khi có đáp án đúng để so khớp trang).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

from source.retrieval.ingest_glue import build_clean_pages  # noqa: E402
from source.retrieval.chunker import chunk_by_page, chunk_fixed_size_fair  # noqa: E402
from source.retrieval.vectorstore import build_index, search  # noqa: E402
from source.qa.qa_generator import generate_answer  # noqa: E402

from script.pilot_tuan6.run_test_qa import (  # noqa: E402
    hit_at_k,
    extract_predicted_pages,
    empty_row,
    HIT_AT_K,
)

STRATEGIES = {
    "fixed_size_fair": chunk_fixed_size_fair,
    "page_aware_no_bridge": partial(chunk_by_page, include_bridge=False),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation #2 — baseline fixed-size công bằng & page-aware không bridge."
    )
    parser.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    parser.add_argument("--test-set", default="data/eval_sets/test_questions.json")
    parser.add_argument("--corpus-dir", default=config.CORPUS_DIR)
    parser.add_argument("--vncorenlp_dir", default="./vncorenlp_models")
    parser.add_argument("--k", type=int, default=config.TOP_K_GENERATION)
    parser.add_argument("--tau", type=float, default=config.TAU)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument(
        "--no-legacy-encoding-normalization",
        action="store_true",
        help="Bỏ qua chuyển mã VNI/TCVN3; mặc định vẫn chuẩn hóa như run_test_qa.py.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    chunk_fn = STRATEGIES[args.strategy]
    out_path = Path(
        args.out or f"results/tuan_bo_sung/ablation_fair_baseline/test_qa_results_{args.strategy}.csv"
    ).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(Path(args.test_set).resolve(), "r", encoding="utf-8") as f:
        test_data = json.load(f)
    all_questions = test_data["questions"]

    questions = [q for q in all_questions if q.get("is_answerable")]
    print(f"*** Loc con {len(questions)}/{len(all_questions)} cau is_answerable=True. ***\n")

    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
        print(f"*** DRY-RUN: chi chay {len(questions)} cau dau. Dung --limit 0 de chay full. ***\n")

    previous_ok: Dict[str, dict] = {}
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        for r in old_rows:
            if str(r.get("is_error", "")).strip().lower() != "true":
                previous_ok[r["id"]] = r
        print(f"*** RESUME: {len(previous_ok)} dong da co san, se bo qua. ***\n")

    corpus_dir = Path(args.corpus_dir).resolve()
    unique_files = sorted({q["source_file"] for q in questions})
    print(f"=== ABLATION #2 - strategy={args.strategy}, k={args.k}, tau={args.tau}, model={args.model} ===")
    print(f"Dung index cho {len(unique_files)} file nguon: {unique_files}\n")

    indexes: Dict[str, object] = {}
    build_errors: Dict[str, str] = {}
    for source_file in unique_files:
        pdf_path = corpus_dir / source_file
        print(f"[Dung index] {source_file} ({args.strategy}) ...")
        try:
            pages = build_clean_pages(
                str(pdf_path),
                normalize_encoding=not args.no_legacy_encoding_normalization,
            )
            chunks = chunk_fn(pages)
            index = build_index(chunks, args.vncorenlp_dir)
            indexes[source_file] = index
            n_bridge = sum(1 for c in chunks if c.get("is_bridge"))
            n_multi_page = sum(1 for c in chunks if c.get("page_range") and not c.get("is_bridge"))
            print(f"  -> {len(pages)} trang, {len(chunks)} chunk ({n_bridge} bridge, "
                  f"{n_multi_page} chunk thuong phu nhieu trang)\n")
        except Exception as e:
            build_errors[source_file] = str(e)
            print(f"  -> LOI dung index (ghi nhan, khong dung script): {e}\n")

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
            hits_at_3 = search(index, q["question"], args.vncorenlp_dir, k=HIT_AT_K)
            hit3 = hit_at_k(hits_at_3, expected_pages, HIT_AT_K)

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
            print(f"  [{done}/{total}] {q['id']:8s} -> LOI KHONG LUONG TRUOC: {e}")

    if not rows:
        print("Khong co dong nao de ghi (0 cau hoi).")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(1 for r in rows if str(r["is_error"]).strip().lower() == "true")
    n_hit3 = sum(1 for r in rows if str(r.get("hit_at_3")).strip().lower() == "true")
    print(f"\nDa ghi {len(rows)} dong vao: {out_path}")
    print(f"Hit@3: {n_hit3}/{len(rows)} | Loi: {n_errors}/{len(rows)}")


if __name__ == "__main__":
    main()

"""Ghi nhận output hiện tại của retrieval + QA trên một vài câu dev.

Script này chỉ xác minh/ghi số liệu, không thay đổi config hoặc logic retrieval.
Mặc định chạy 3 câu đầu của dev_questions_normalized.json với chunking
page_aware và ghi answer_text, citations cùng toàn bộ điểm của các hit.

Cách chạy:
    python script/pilot_tuan6/verify_vectorstore_current.py

Chọn câu hỏi cụ thể:
    python script/pilot_tuan6/verify_vectorstore_current.py --ids dev_01 dev_05 dev_13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from source.qa.qa_generator import generate_answer
from source.retrieval.chunker import chunk_by_page
from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.vectorstore import ChunkIndex, build_index, search


def _hit_to_dict(hit) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "source_file": hit.source_file,
        "page_number": hit.page_number,
        "page_range": hit.page_range,
        "is_bridge": hit.is_bridge,
        "score": hit.score,
        "bm25_score": hit.bm25_score,
        "rrf_score": hit.rrf_score,
        "text": hit.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ghi nhận output hiện tại của vectorstore + QA trên dev set."
    )
    parser.add_argument("--dev-set", default=config.DEV_SET_PATH)
    parser.add_argument("--corpus-dir", default=config.CORPUS_DIR)
    parser.add_argument("--vncorenlp-dir", default="vncorenlp_models")
    parser.add_argument("--k", type=int, default=config.TOP_K_GENERATION)
    parser.add_argument("--tau", type=float, default=config.TAU)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--ids", nargs="+", help="ID câu hỏi; ghi đè --limit nếu có")
    parser.add_argument(
        "--out",
        default="results/tuan6_pilot/vectorstore_current_verification.json",
    )
    args = parser.parse_args()

    dev_set_path = Path(args.dev_set).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    vncorenlp_dir = str(Path(args.vncorenlp_dir).resolve())
    out_path = Path(args.out).resolve()

    with dev_set_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)["questions"]

    if args.ids:
        by_id = {question["id"]: question for question in questions}
        missing_ids = [question_id for question_id in args.ids if question_id not in by_id]
        if missing_ids:
            raise SystemExit(f"Không tìm thấy question ID: {', '.join(missing_ids)}")
        questions = [by_id[question_id] for question_id in args.ids]
    else:
        questions = questions[: args.limit]

    if not questions:
        raise SystemExit("Không có câu hỏi để chạy.")

    indexes: Dict[str, ChunkIndex] = {}
    for source_file in sorted({question["source_file"] for question in questions}):
        pages = build_clean_pages(str(corpus_dir / source_file))
        chunks = chunk_by_page(pages)
        indexes[source_file] = build_index(chunks, vncorenlp_dir)

    rows: List[dict] = []
    for question in questions:
        hits = search(
            indexes[question["source_file"]],
            question["question"],
            vncorenlp_dir,
            k=args.k,
        )
        answer = generate_answer(
            question["question"],
            hits,
            tau=args.tau,
            target_model=args.model,
            corpus_dir=str(corpus_dir),
        )
        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "source_file": question["source_file"],
                "config": "page_aware",
                "k": args.k,
                "tau": args.tau,
                "model": args.model,
                "answer_text": answer.answer_text,
                "citations": answer.citations,
                "is_abstained": answer.is_abstained,
                "abstain_reason": answer.abstain_reason,
                "is_error": answer.is_error,
                "error_message": answer.error_message,
                "retrieval_hits": [_hit_to_dict(hit) for hit in hits],
            }
        )
        print(f"{question['id']}: ghi nhận {len(hits)} retrieval hits")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Đã ghi số liệu hiện tại vào: {out_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.retrieval.chunker import chunk_by_page
from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.vectorstore import build_index

TEST_SET_PATH = PROJECT_ROOT / "data" / "eval_sets" / "test_questions.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "tuan_bo_sung" / "ablation" / "ablation_hybrid_results.csv"
VNCORENLP_DIR = PROJECT_ROOT / "vncorenlp_models"


def load_hit_at_k_helper():
    """Tải helper hit_at_k đang dùng trong project để giữ logic matching expected_page nguyên bản."""
    helper_path = PROJECT_ROOT / "script" / "pilot_tuan6" / "run_test_qa.py"
    spec = importlib.util.spec_from_file_location("run_test_qa_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp helper từ {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hit_at_k


HIT_AT_K_HELPER = load_hit_at_k_helper()


def load_test_questions(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError(f"Không có câu hỏi nào trong {path}")
    return questions


def build_index_for_source(source_file: str, vncorenlp_dir: str):
    pdf_path = PROJECT_ROOT / "data" / "corpus" / source_file
    pages = build_clean_pages(str(pdf_path))
    chunks = chunk_by_page(pages)
    return build_index(chunks, str(vncorenlp_dir))


def format_retrieved_pages(hit) -> List[int | str]:
    values: List[int | str] = []
    for item in hit:
        if item.page_number is not None:
            values.append(int(item.page_number))
        elif item.page_range:
            values.append(item.page_range)
        else:
            values.append("unknown")
    return values


def main() -> None:
    questions = load_test_questions(TEST_SET_PATH)
    indexes: Dict[str, object] = {}

    rows: List[dict] = []
    summary: Dict[str, List[int]] = {
        "hybrid": [],
        "dense_only": [],
        "bm25_only": [],
    }

    for q in questions:
        source_file = q["source_file"]
        if source_file not in indexes:
            indexes[source_file] = build_index_for_source(source_file, str(VNCORENLP_DIR))

        index = indexes[source_file]
        expected_pages = [int(p) for p in (q.get("expected_page") or [])]

        if q.get("is_answerable") is not True:
            continue

        configs = [
            ("hybrid", {"use_dense": True, "use_bm25": True}),
            ("dense_only", {"use_dense": True, "use_bm25": False}),
            ("bm25_only", {"use_dense": False, "use_bm25": True}),
        ]

        for config_name, kwargs in configs:
            hits = index.search_hybrid(q["question"], k=3, **kwargs)
            hit_at_3 = 1 if HIT_AT_K_HELPER(hits, expected_pages, 3) else 0
            summary[config_name].append(hit_at_3)

            retrieved_chunk_ids = [h.chunk_id for h in hits[:3]]
            retrieved_pages = format_retrieved_pages(hits[:3])

            rows.append(
                {
                    "question_id": q["id"],
                    "config": config_name,
                    "hit_at_3": hit_at_3,
                    "retrieved_chunk_ids": json.dumps(retrieved_chunk_ids, ensure_ascii=False),
                    "retrieved_pages": json.dumps(retrieved_pages, ensure_ascii=False),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "config", "hit_at_3", "retrieved_chunk_ids", "retrieved_pages"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Hit@3 trung bình theo cấu hình")
    print(f"{'config':<12} {'hit@3':>8} {'n':>4}")
    for config_name in ["hybrid", "dense_only", "bm25_only"]:
        values = summary[config_name]
        avg = (sum(values) / len(values)) if values else 0.0
        print(f"{config_name:<12} {avg * 100:>7.1f}% {len(values):>4}")

    print(f"\nCSV saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from source.retrieval.chunker import chunk_by_page, chunk_fixed_size
from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.vectorstore import build_index, search


TEST_SET_PATH = PROJECT_ROOT / "data/eval_sets/test_questions.json"
CORPUS_DIR = PROJECT_ROOT / config.CORPUS_DIR
VNCORENLP_DIR = PROJECT_ROOT / "vncorenlp_models"
OUTPUT_PATH = PROJECT_ROOT / "results/tuan9_tau_sweep/test_retrieval_scores.csv"
RESULT_PATHS = {
    "page_aware": PROJECT_ROOT / "results/tuan8/test_qa_results_page_aware_50cau.csv",
    "fixed_size": PROJECT_ROOT / "results/tuan8/test_qa_results_fixed_size_50cau.csv",
}
STRATEGIES = {
    "page_aware": chunk_by_page,
    "fixed_size": chunk_fixed_size,
}
FIELDNAMES = ["id", "config", "max_score", "is_answerable"]


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {OUTPUT_PATH}")

    for path in (TEST_SET_PATH, *RESULT_PATHS.values()):
        if not path.is_file():
            raise SystemExit(f"Required input file not found: {path}")

    with TEST_SET_PATH.open("r", encoding="utf-8") as json_file:
        questions = json.load(json_file)["questions"]

    score_rows: List[dict] = []
    scores_by_key: Dict[Tuple[str, str], float] = {}

    for config_name, chunk_function in STRATEGIES.items():
        source_files = sorted({question["source_file"] for question in questions})
        indexes = {}

        for source_file in source_files:
            pages = build_clean_pages(
                str(CORPUS_DIR / source_file),
                normalize_encoding=True,
            )
            chunks = chunk_function(pages)
            indexes[source_file] = build_index(chunks, str(VNCORENLP_DIR))

        for question in questions:
            hits = search(
                indexes[question["source_file"]],
                question["question"],
                str(VNCORENLP_DIR),
                k=config.TOP_K_GENERATION,
            )
            if not hits:
                raise RuntimeError(
                    f"Search returned no hits for {question['id']} ({config_name})"
                )

            max_score = max(float(hit.score) for hit in hits)
            key = (question["id"], config_name)
            scores_by_key[key] = max_score
            score_rows.append(
                {
                    "id": question["id"],
                    "config": config_name,
                    "max_score": max_score,
                    "is_answerable": question["is_answerable"],
                }
            )

    gate_rows: Dict[str, List[Tuple[bool, str, float, str]]] = {
        config_name: [] for config_name in STRATEGIES
    }
    for config_name, result_path in RESULT_PATHS.items():
        seen_keys = set()
        for result in read_csv_rows(result_path):
            result_config = result.get("config", "")
            if result_config != config_name:
                raise RuntimeError(
                    f"Unexpected config in {result_path}: {result_config!r}"
                )
            key = (result["id"], result_config)
            if key in seen_keys:
                raise RuntimeError(f"Duplicate result row: {key}")
            seen_keys.add(key)
            if key not in scores_by_key:
                raise RuntimeError(f"No computed score for result row: {key}")

            max_score = scores_by_key[key]
            abstain_reason = result.get("abstain_reason", "") or ""
            if abstain_reason == "retrieval_threshold":
                matches = max_score < config.TAU
            else:
                matches = max_score >= config.TAU
            gate_rows[config_name].append(
                (matches, result["id"], max_score, abstain_reason)
            )

    if OUTPUT_PATH.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(score_rows)

    for config_name, rows in gate_rows.items():
        matched = sum(1 for row in rows if row[0])
        mismatched = len(rows) - matched
        print(f"GATE {config_name}: KHỚP={matched} KHÔNG_KHỚP={mismatched}")
        for matches, question_id, max_score, abstain_reason in rows:
            if not matches:
                print(
                    f"KHÔNG_KHỚP id={question_id} config={config_name} "
                    f"max_score={max_score} abstain_reason={abstain_reason}"
                )

    print(f"CSV_PATH={OUTPUT_PATH}")
    print("CSV_BEGIN")
    print(OUTPUT_PATH.read_text(encoding="utf-8"), end="")
    print("CSV_END")


if __name__ == "__main__":
    main()
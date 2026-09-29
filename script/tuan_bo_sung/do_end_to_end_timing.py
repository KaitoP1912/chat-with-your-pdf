"""Cold-process end-to-end timing for the three locked QA configurations.

Each run is a fresh Python subprocess. Imports are timed separately; the
total timer starts after imports, before PDF loading, and ends after the
first answer is returned. build_clean_pages() loads the PDF internally, so
total_excl_duplicate_load_seconds subtracts the separately timed first read.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_NAME = "normal_hienphap_33tr.pdf"
TEST_SET_PATH = PROJECT_ROOT / "data" / "eval_sets" / "test_questions.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "tuan_bo_sung" / "end_to_end_timing" / "end_to_end_timing.csv"
CONFIGS = ("page_aware", "fixed_size", "longcontext")
RUNS_PER_CONFIG = 3
RESULT_MARKER = "RESULT_JSON:"

STAGE_COLUMNS = (
    "load_pdf_pages_seconds",
    "detect_scan_seconds",
    "build_clean_pages_seconds",
    "chunk_seconds",
    "build_index_seconds",
    "search_seconds",
    "generate_answer_seconds",
    "build_prompt_seconds",
    "generate_content_seconds",
)
FIELDNAMES = (
    "test_id",
    "config",
    "run",
    "import_seconds",
    *STAGE_COLUMNS,
    "total_seconds",
    "total_excl_duplicate_load_seconds",
    "retry_count",
    "is_abstained",
    "error_message",
)


def _timed(
    stage_name: str,
    timings: dict[str, float],
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings[stage_name] = time.perf_counter() - started


def _load_question() -> dict[str, Any]:
    with TEST_SET_PATH.open("r", encoding="utf-8") as file:
        test_data = json.load(file)
    matches = [
        question
        for question in test_data["questions"]
        if question.get("id") == "test_01"
    ]
    if len(matches) != 1:
        raise ValueError("Test Set phải có đúng một câu test_01.")
    question = matches[0]
    if question.get("source_file") != PDF_NAME or not question.get("is_answerable"):
        raise ValueError("test_01 không phải câu answerable của PDF đã chọn.")
    return question


def _run_worker(config_name: str, run_number: int) -> None:
    started_import = time.perf_counter()
    sys.path.insert(0, str(PROJECT_ROOT))

    import config
    from source.ingestion.pdf_loader import load_pdf_pages
    from source.ingestion.scan_detector import detect_scan
    from source.retrieval.ingest_glue import build_clean_pages

    if config_name in ("page_aware", "fixed_size"):
        from source.qa.qa_generator import generate_answer
        from source.retrieval.chunker import chunk_by_page, chunk_fixed_size
        from source.retrieval.vectorstore import build_index, search
        longcontext = None
    else:
        from script.tuan_bo_sung import run_longcontext_used_sources as longcontext

        generate_answer = None
        chunk_by_page = None
        chunk_fixed_size = None
        build_index = None
        search = None

    import_seconds = time.perf_counter() - started_import
    k = config.TOP_K_GENERATION
    tau = config.TAU
    model_name = config.MODEL_NAME
    timings: dict[str, float] = {}
    started_total = time.perf_counter()
    question = _load_question()
    pdf_path = PROJECT_ROOT / "data" / "corpus" / PDF_NAME
    is_abstained: bool | None = None
    error_message = ""
    retry_count: int | None = None

    try:
        if config_name in ("page_aware", "fixed_size"):
            raw_pages = _timed("load_pdf_pages_seconds", timings, load_pdf_pages, str(pdf_path))
            _timed("detect_scan_seconds", timings, detect_scan, raw_pages)
            clean_pages = _timed(
                "build_clean_pages_seconds", timings, build_clean_pages, str(pdf_path)
            )
            chunk_function = chunk_by_page if config_name == "page_aware" else chunk_fixed_size
            chunks = _timed("chunk_seconds", timings, chunk_function, clean_pages)
            vncorenlp_dir = str(PROJECT_ROOT / "vncorenlp_models")
            index = _timed(
                "build_index_seconds", timings, build_index, chunks, vncorenlp_dir
            )
            hits = _timed(
                "search_seconds", timings, search, index, question["question"], vncorenlp_dir, k
            )
            answer = _timed(
                "generate_answer_seconds",
                timings,
                generate_answer,
                question["question"],
                hits,
                tau=tau,
                target_model=model_name,
                corpus_dir=str(pdf_path.parent),
            )
            is_abstained = bool(answer.is_abstained)
            retry_count = getattr(answer, "retry_count", None)
        else:
            _timed("load_pdf_pages_seconds", timings, load_pdf_pages, str(pdf_path))
            clean_pages = _timed(
                "build_clean_pages_seconds", timings, build_clean_pages, str(pdf_path)
            )
            def build_full_text_prompt() -> str:
                document_block, _page_lookup = longcontext.build_document_block_and_lookup(clean_pages)
                return longcontext.PROMPT_TEMPLATE.format(
                    abstain_text=config.MODEL_ABSTAIN_TEXT,
                    document_block=document_block,
                    question=question["question"],
                )

            prompt = _timed("build_prompt_seconds", timings, build_full_text_prompt)

            retry_count = 0

            def generate_longcontext_content() -> str:
                nonlocal retry_count
                longcontext._ensure_configured()
                model = longcontext.genai.GenerativeModel(
                    model_name,
                    generation_config={"temperature": config.GENERATION_TEMPERATURE},
                )
                while True:
                    try:
                        return model.generate_content(prompt).text
                    except Exception as error:
                        if (
                            longcontext._is_rate_limit_error(error)
                            and retry_count < longcontext.MAX_RATE_LIMIT_RETRIES
                        ):
                            retry_count += 1
                            time.sleep(longcontext.RATE_LIMIT_RETRY_SECONDS)
                            continue
                        raise

            response_text = _timed(
                "generate_content_seconds", timings, generate_longcontext_content
            )
            parsed = longcontext._parse_json_response(response_text)
            answer_text = parsed.get("answer", response_text) if isinstance(parsed, dict) else response_text
            is_abstained = str(answer_text).strip() == config.MODEL_ABSTAIN_TEXT
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"

    total_seconds = time.perf_counter() - started_total
    result = {
        "test_id": question["id"],
        "config": config_name,
        "run": run_number,
        "import_seconds": import_seconds,
        **{column: timings.get(column, "") for column in STAGE_COLUMNS},
        "total_seconds": total_seconds,
        "total_excl_duplicate_load_seconds": (
            total_seconds - timings["load_pdf_pages_seconds"]
            if "load_pdf_pages_seconds" in timings
            else ""
        ),
        "retry_count": retry_count if retry_count is not None else "",
        "is_abstained": is_abstained if is_abstained is not None else "",
        "error_message": error_message,
    }
    print(f"{RESULT_MARKER}{json.dumps(result, ensure_ascii=False)}", flush=True)


def _run_parent() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    import config

    question = _load_question()
    print(f"Câu hỏi dùng chung: {question['id']} ({question['source_file']})")
    print(
        f"Cấu hình khóa: k={config.TOP_K_GENERATION}, tau={config.TAU}, "
        f"model={config.MODEL_NAME}; {RUNS_PER_CONFIG} lần/config."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        output_file.flush()

        total_workers = len(CONFIGS) * RUNS_PER_CONFIG
        completed_workers = 0
        for config_name in CONFIGS:
            for run_number in range(1, RUNS_PER_CONFIG + 1):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    config_name,
                    str(run_number),
                ]
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                marker_lines = [
                    line for line in completed.stdout.splitlines()
                    if line.startswith(RESULT_MARKER)
                ]
                if completed.returncode != 0 or not marker_lines:
                    print(completed.stdout, end="")
                    print(completed.stderr, file=sys.stderr, end="")
                    result = {
                        "test_id": question["id"],
                        "config": config_name,
                        "run": run_number,
                        "import_seconds": "",
                        **{column: "" for column in STAGE_COLUMNS},
                        "total_seconds": "",
                        "total_excl_duplicate_load_seconds": "",
                        "retry_count": "",
                        "is_abstained": "",
                        "error_message": f"Worker exited with code {completed.returncode}.",
                    }
                else:
                    result = json.loads(marker_lines[-1][len(RESULT_MARKER):])
                    if completed.stderr:
                        print(completed.stderr, file=sys.stderr, end="")

                writer.writerow(result)
                output_file.flush()
                status = "error" if result["error_message"] else str(result["is_abstained"])
                print(
                    f"{config_name} run {run_number}: "
                    f"total={result['total_seconds']}s, "
                    f"is_abstained={status}"
                )
                completed_workers += 1
                if completed_workers < total_workers:
                    time.sleep(5)

    print(f"Kết quả: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "RUN"))
    args = parser.parse_args()
    if args.worker:
        config_name, run_number = args.worker
        if config_name not in CONFIGS:
            raise ValueError(f"Config không hợp lệ: {config_name}")
        _run_worker(config_name, int(run_number))
    else:
        _run_parent()


if __name__ == "__main__":
    main()
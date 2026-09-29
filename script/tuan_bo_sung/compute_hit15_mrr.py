from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUTS = {
    "page_aware": PROJECT_ROOT / "results" / "tuan_bo_sung" / "rerun_full" / "test_qa_results_page_aware.csv",
    "fixed_size": PROJECT_ROOT / "results" / "tuan_bo_sung" / "rerun_full" / "test_qa_results_fixed_size.csv",
}


def parse_expected_pages(raw: str) -> set[int]:
    if raw is None or raw == "":
        return set()
    pages = set()
    for part in str(raw).split(";"):
        token = part.strip()
        if not token:
            continue
        try:
            pages.add(int(token))
        except ValueError:
            continue
    return pages


def parse_top15_pages(raw: str) -> List[str]:
    if raw is None or raw == "":
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x) for x in parsed]


def normalize_range_token(token: str) -> List[int]:
    if token is None:
        return []
    token = str(token).strip()
    if not token:
        return []
    if "-" in token:
        try:
            a, b = token.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            return [a_i, b_i]
        except ValueError:
            return []
    try:
        return [int(token)]
    except ValueError:
        return []


def evaluate_hit_and_rr(row: dict) -> Tuple[bool, float]:
    expected_pages = parse_expected_pages(row.get("expected_page", ""))
    if not expected_pages:
        return False, 0.0

    top15_pages = parse_top15_pages(row.get("top15_pages", "[]"))
    if not top15_pages:
        return False, 0.0

    first_rank = 0
    hit = False
    for rank, page_token in enumerate(top15_pages, start=1):
        page_values = normalize_range_token(page_token)
        if not page_values:
            continue
        if any(v in expected_pages for v in page_values):
            first_rank = rank
            hit = True
            break

    rr = 1.0 / first_rank if hit else 0.0
    return hit, rr


def load_answerable_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("is_answerable", "")).strip().lower() == "true":
                rows.append(row)
    return rows


def summarize(rows: Iterable[dict]) -> Tuple[int, float, float]:
    rows = list(rows)
    total = len(rows)
    hits = 0
    rr_sum = 0.0
    for row in rows:
        hit, rr = evaluate_hit_and_rr(row)
        if hit:
            hits += 1
        rr_sum += rr
    hit_rate = hits / total if total else 0.0
    mrr = rr_sum / total if total else 0.0
    return hits, hit_rate, mrr


def main() -> None:
    print(f"{'config':<12} {'Hit@15':>16} {'MRR':>12}")
    print("-" * 42)

    for config_name, path in INPUTS.items():
        rows = load_answerable_rows(path)
        hits, hit_rate, mrr = summarize(rows)
        print(
            f"{config_name:<12} "
            f"{hits}/{len(rows):<2} {hit_rate * 100:>5.1f}% "
            f"{mrr:>11.4f}"
        )


if __name__ == "__main__":
    main()

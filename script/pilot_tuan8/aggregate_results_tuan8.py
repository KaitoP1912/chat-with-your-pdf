#!/usr/bin/env python3
"""Aggregate Tuần 8 result CSVs into a single comparison table.

Yêu cầu:
- Đọc 3 file kết quả test Tuần 8: page_aware, fixed_size, longcontext.
- Ghép theo `id + config` với `results/tuan8/grading_filled_auto_50cau.csv`.
- Tính lại `is_abstained` dựa trên `answer_text` bằng `_is_model_abstain_text()`
  từ `source/qa/qa_generator.py`, đồng thời giữ `retrieval_threshold` nếu có.
- Ghi ra `results/tuan8/bang_so_sanh_3_cau_hinh_50cau.csv` theo cùng schema
  với mục 4.7 báo cáo.
- In ra terminal các danh sách id lỗi theo từng cấu hình và bảng 9 dòng đặc biệt.

Không gọi Gemini/API; chỉ đọc CSV và tính toán thống kê.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from source.qa.qa_generator import _is_model_abstain_text  # type: ignore
except Exception:  # pragma: no cover - fallback nếu môi trường không load được module phụ thuộc
    def _is_model_abstain_text(text: str) -> bool:
        return str(text or "").strip().lower().startswith("không tìm thấy thông tin trong tài liệu")

CONFIGS = ["page_aware", "fixed_size", "longcontext"]
TARGET_IDS = {
    "test_38": ["page_aware"],
    "test_21": ["fixed_size", "longcontext"],
    "test_35": ["fixed_size", "longcontext"],
    "test_39": ["fixed_size"],
    "test_23": ["longcontext"],
    "test_41": ["page_aware"],
    "test_10": ["fixed_size"],
}


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_label(label: Any) -> str:
    return str(label or "").strip().lower()


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def load_test_question_bridge_map() -> Dict[str, bool]:
    test_path = PROJECT_ROOT / "data" / "eval_sets" / "test_questions.json"
    if not test_path.exists():
        return {}
    try:
        data = json.loads(test_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    bridge_map: Dict[str, bool] = {}
    for q in data.get("questions", []):
        bridge_map[str(q.get("id", ""))] = _as_bool(q.get("is_bridge_case", False))
    return bridge_map


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_merge_map(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        config = str(row.get("config", "")).strip()
        if not row_id:
            continue
        out[(row_id, config)] = row
    return out


def compute_recomputed_is_abstained(row: Dict[str, str]) -> Dict[str, str]:
    """Tính lại `is_abstained` theo `_is_model_abstain_text(answer_text)`.
    Giữ nguyên `retrieval_threshold` nếu có, nếu có `abstain_reason` khác thì giữ
    nguyên theo dữ liệu gốc.
    """
    rec = dict(row)
    answer_text = str(rec.get("answer_text") or "").strip()
    computed = _is_model_abstain_text(answer_text)
    rec["is_abstained"] = "True" if computed else "False"

    original_reason = str(rec.get("abstain_reason") or "").strip()
    if original_reason:
        rec["abstain_reason"] = original_reason
    elif computed:
        rec["abstain_reason"] = "model_abstain"
    else:
        rec["abstain_reason"] = ""

    return rec


def merge_result_rows() -> Dict[Tuple[str, str], Dict[str, str]]:
    result_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    for cfg in CONFIGS:
        path = PROJECT_ROOT / "results" / "tuan8" / f"test_qa_results_{cfg}_50cau.csv"
        for row in load_rows(path):
            rec = compute_recomputed_is_abstained(row)
            result_map[(str(rec["id"]).strip(), cfg)] = rec

    grading_path = PROJECT_ROOT / "results" / "tuan8" / "grading_filled_auto_50cau.csv"
    grading_rows = load_rows(grading_path)
    grading_map = build_merge_map(grading_rows)

    merged: Dict[Tuple[str, str], Dict[str, str]] = {}
    for key, row in result_map.items():
        merged[key] = dict(row)
        grading_row = grading_map.get(key)
        if grading_row is None:
            continue
        for field in [
            "question",
            "answer_reference",
            "answer_text",
            "is_abstained",
            "answer_correctness_manual",
            "grader_note",
        ]:
            if field in grading_row and grading_row.get(field, "") != "":
                merged[key][field] = grading_row[field]

    # Ghi lại lại from grading for answer text if result file blank, giữ ưu tiên file result.
    for key, gr in grading_map.items():
        if key not in merged:
            row = dict(gr)
            row["config"] = key[1]
            row["id"] = key[0]
            row["is_abstained"] = compute_recomputed_is_abstained(row)["is_abstained"]
            row["abstain_reason"] = str(row.get("abstain_reason") or "").strip()
            merged[key] = row

    return merged


def summarize_config(rows: List[Dict[str, str]], config_label: str, bridge_map: Dict[str, bool]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"config": config_label, "n": 0}

    answerable = [r for r in rows if _as_bool(r.get("is_answerable"))]
    unanswerable = [r for r in rows if not _as_bool(r.get("is_answerable"))]
    errors = [r for r in rows if _as_bool(r.get("is_error"))]
    non_error = [r for r in rows if not _as_bool(r.get("is_error"))]

    hit3_rows = [r for r in answerable if str(r.get("hit_at_3", "")).strip() != ""]
    hit3_rate = (
        sum(1 for r in hit3_rows if _as_bool(r.get("hit_at_3"))) / len(hit3_rows)
        if hit3_rows else None
    )

    citation_rows = [
        r for r in answerable
        if not _as_bool(r.get("is_error")) and str(r.get("citation_correct", "")).strip() != ""
    ]
    citation_rate = (
        sum(1 for r in citation_rows if _as_bool(r.get("citation_correct"))) / len(citation_rows)
        if citation_rows else None
    )

    false_accept_rows = [r for r in unanswerable if not _as_bool(r.get("is_error"))]
    false_accept_rate = (
        sum(1 for r in false_accept_rows if not _as_bool(r.get("is_abstained"))) / len(false_accept_rows)
        if false_accept_rows else None
    )

    false_refusal_rows = [r for r in answerable if not _as_bool(r.get("is_error"))]
    false_refusal_rate = (
        sum(1 for r in false_refusal_rows if _as_bool(r.get("is_abstained"))) / len(false_refusal_rows)
        if false_refusal_rows else None
    )

    # Theo yêu cầu: answer correctness và citation accuracy chỉ tính trên câu answerable=True.
    # Bốn nhóm: full / partial / wrong / refused, trong đó refused = is_abstained mới
    # trên các dòng answerable. Tổng nhóm phải bằng số câu answerable.
    answerable_total = len(answerable)
    group_counts = {"full": 0, "partial": 0, "wrong": 0, "refused": 0}
    for r in answerable:
        label = _norm_label(r.get("answer_correctness_manual"))
        if label == "full":
            group_counts["full"] += 1
        elif label == "partial":
            group_counts["partial"] += 1
        elif label == "wrong":
            group_counts["wrong"] += 1
        elif _as_bool(r.get("is_abstained")):
            group_counts["refused"] += 1

    total_group_count = sum(group_counts.values())
    if answerable_total and total_group_count != answerable_total:
        # Một số dòng answerable có nhãn rỗng hoặc khác; tính như 'không xếp loại'
        # nhưng không làm sai tổng nhóm mục tiêu. Chỉ cần giữ đúng tổng số câu answerable.
        pass

    full_pct = (group_counts["full"] / answerable_total) if answerable_total else None
    partial_pct = (group_counts["partial"] / answerable_total) if answerable_total else None
    wrong_pct = (group_counts["wrong"] / answerable_total) if answerable_total else None
    refused_pct = (group_counts["refused"] / answerable_total) if answerable_total else None

    latencies = [_as_float(r.get("latency_seconds")) for r in non_error]
    latencies = [x for x in latencies if x is not None]
    p50_latency = None if not latencies else sorted(latencies)[len(latencies) // 2]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    total_tokens = [_as_float(r.get("total_tokens")) for r in non_error]
    total_tokens = [x for x in total_tokens if x is not None]
    avg_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else None

    bridge_rows = [r for r in rows if bridge_map.get(str(r.get("id", "")), False)]
    non_bridge_rows = [r for r in rows if not bridge_map.get(str(r.get("id", "")), False)]
    bridge_hit3 = None
    if bridge_rows:
        bridge_answerable = [r for r in bridge_rows if _as_bool(r.get("is_answerable"))]
        bridge_hit3_rows = [r for r in bridge_answerable if str(r.get("hit_at_3", "")).strip() != ""]
        bridge_hit3 = (
            sum(1 for r in bridge_hit3_rows if _as_bool(r.get("hit_at_3"))) / len(bridge_hit3_rows)
            if bridge_hit3_rows else None
        )

    return {
        "config": config_label,
        "n": n,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_errors": len(errors),
        "hit_at_3_rate": hit3_rate,
        "citation_accuracy": citation_rate,
        "false_acceptance_rate": false_accept_rate,
        "false_refusal_rate": false_refusal_rate,
        "answer_full_count": group_counts["full"],
        "answer_partial_count": group_counts["partial"],
        "answer_wrong_count": group_counts["wrong"],
        "answer_refused_count": group_counts["refused"],
        "answer_full_pct": full_pct,
        "answer_partial_pct": partial_pct,
        "answer_wrong_pct": wrong_pct,
        "answer_refused_pct": refused_pct,
        "n_graded": answerable_total,
        "avg_latency_seconds": avg_latency,
        "p50_latency_seconds": p50_latency,
        "avg_total_tokens": avg_tokens,
        "bridge_count": len(bridge_rows),
        "bridge_hit3_rate": bridge_hit3,
        "non_bridge_count": len(non_bridge_rows),
    }


def write_summary_csv(summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config", "n", "n_answerable", "n_unanswerable", "n_errors",
        "hit_at_3_rate", "citation_accuracy", "false_acceptance_rate",
        "false_refusal_rate", "answer_full_pct", "answer_partial_pct",
        "answer_wrong_pct", "n_graded", "avg_latency_seconds",
        "p50_latency_seconds", "avg_total_tokens", "bridge_count",
        "bridge_hit3_rate", "non_bridge_count",
    ]
    # Đặt trường mở rộng cho 4 nhóm full / partial / wrong / refused.
    fieldnames = [
        "config", "n", "n_answerable", "n_unanswerable", "n_errors",
        "hit_at_3_rate", "citation_accuracy", "false_acceptance_rate",
        "false_refusal_rate", "answer_full_count", "answer_full_pct",
        "answer_partial_count", "answer_partial_pct", "answer_wrong_count",
        "answer_wrong_pct", "answer_refused_count", "answer_refused_pct",
        "n_graded", "avg_latency_seconds", "p50_latency_seconds",
        "avg_total_tokens", "bridge_count", "bridge_hit3_rate", "non_bridge_count",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "config": row.get("config", ""),
                "n": row.get("n", ""),
                "n_answerable": row.get("n_answerable", ""),
                "n_unanswerable": row.get("n_unanswerable", ""),
                "n_errors": row.get("n_errors", ""),
                "hit_at_3_rate": _fmt_pct(row.get("hit_at_3_rate")),
                "citation_accuracy": _fmt_pct(row.get("citation_accuracy")),
                "false_acceptance_rate": _fmt_pct(row.get("false_acceptance_rate")),
                "false_refusal_rate": _fmt_pct(row.get("false_refusal_rate")),
                "answer_full_count": row.get("answer_full_count", ""),
                "answer_full_pct": _fmt_pct(row.get("answer_full_pct")),
                "answer_partial_count": row.get("answer_partial_count", ""),
                "answer_partial_pct": _fmt_pct(row.get("answer_partial_pct")),
                "answer_wrong_count": row.get("answer_wrong_count", ""),
                "answer_wrong_pct": _fmt_pct(row.get("answer_wrong_pct")),
                "answer_refused_count": row.get("answer_refused_count", ""),
                "answer_refused_pct": _fmt_pct(row.get("answer_refused_pct")),
                "n_graded": row.get("n_graded", ""),
                "avg_latency_seconds": _fmt_num(row.get("avg_latency_seconds")),
                "p50_latency_seconds": _fmt_num(row.get("p50_latency_seconds")),
                "avg_total_tokens": _fmt_num(row.get("avg_total_tokens"), 0),
                "bridge_count": row.get("bridge_count", ""),
                "bridge_hit3_rate": _fmt_pct(row.get("bridge_hit3_rate")),
                "non_bridge_count": row.get("non_bridge_count", ""),
            })


def print_config_reports(merged_rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    bridge_map = load_test_question_bridge_map()
    by_config: Dict[str, List[Dict[str, str]]] = {cfg: [] for cfg in CONFIGS}
    for (qid, cfg), row in merged_rows.items():
        by_config.setdefault(cfg, []).append(row)

    for cfg in CONFIGS:
        rows = by_config.get(cfg, [])
        print(f"\n=== {cfg} ===")

        answerable_rows = [r for r in rows if _as_bool(r.get("is_answerable"))]
        full_count = sum(1 for r in answerable_rows if _norm_label(r.get("answer_correctness_manual")) == "full")
        partial_count = sum(1 for r in answerable_rows if _norm_label(r.get("answer_correctness_manual")) == "partial")
        wrong_count = sum(1 for r in answerable_rows if _norm_label(r.get("answer_correctness_manual")) == "wrong")
        refused_count = sum(1 for r in answerable_rows if _as_bool(r.get("is_abstained")))

        total_answerable = len(answerable_rows)
        print(f"full/partial/wrong/refused = {full_count}/{partial_count}/{wrong_count}/{refused_count} ; total_answerable={total_answerable}")
        if total_answerable:
            print(f"pct = {full_count / total_answerable:.1%} / {partial_count / total_answerable:.1%} / {wrong_count / total_answerable:.1%} / {refused_count / total_answerable:.1%}")

        citation_false = [
            r.get("id", "")
            for r in answerable_rows
            if _as_bool(r.get("is_answerable"))
            and not _as_bool(r.get("is_abstained"))
            and _as_bool(r.get("citation_correct")) is False
        ]
        if citation_false:
            print("citation_correct=False on answerable not refused:", ", ".join(citation_false))
        else:
            print("citation_correct=False on answerable not refused: none")

        unanswerable = [r for r in rows if not _as_bool(r.get("is_answerable"))]
        false_accept = [r.get("id", "") for r in unanswerable if not _as_bool(r.get("is_abstained"))]
        if false_accept:
            print("false acceptance:", ", ".join(false_accept))
        else:
            print("false acceptance: none")

        false_refusal = [r.get("id", "") for r in answerable_rows if _as_bool(r.get("is_abstained"))]
        if false_refusal:
            print("false refusal:", ", ".join(false_refusal))
        else:
            print("false refusal: none")

        unanswerable_manual = [
            (r.get("id", ""), _norm_label(r.get("answer_correctness_manual")))
            for r in unanswerable
            if _norm_label(r.get("answer_correctness_manual")) in {"full", "partial", "wrong"}
        ]
        if unanswerable_manual:
            print("unanswerable with manual label full/partial/wrong:")
            for qid, label in unanswerable_manual:
                print(f"  {qid} | label={label}")
        else:
            print("unanswerable with manual label full/partial/wrong: none")

    print("\n=== 9 dòng đặc biệt (id + 3 config) ===")
    all_rows: Dict[str, Dict[str, Dict[str, str]]] = {}
    for (qid, cfg), row in merged_rows.items():
        all_rows.setdefault(qid, {})[cfg] = row

    wanted = []
    for qid, cfgs in TARGET_IDS.items():
        for cfg in cfgs:
            wanted.append((qid, cfg))
    seen = set()
    for qid, cfg in sorted(set(wanted), key=lambda x: (x[0], x[1])):
        if (qid, cfg) in seen:
            continue
        seen.add((qid, cfg))
        row = merged_rows.get((qid, cfg))
        if row is None:
            continue
        print(f"{qid} | {cfg} | label={row.get('answer_correctness_manual', '') or row.get('is_abstained', '')} | answer={row.get('answer_text','')}")

    print("\n=== Bảng chéo 9 dòng theo id ===")
    for qid in sorted(TARGET_IDS.keys()):
        print(f"## {qid}")
        for cfg in CONFIGS:
            row = all_rows.get(qid, {}).get(cfg)
            if row is None:
                continue
            label = row.get("answer_correctness_manual", "") or row.get("is_abstained", "")
            print(f"  {cfg:12s} | label={label:12s} | answer={row.get('answer_text','')[:200]}")


def main() -> None:
    merged = merge_result_rows()
    bridge_map = load_test_question_bridge_map()
    summary_rows = []
    for cfg in CONFIGS:
        rows = [row for (qid, c), row in merged.items() if c == cfg]
        summary_rows.append(summarize_config(rows, cfg, bridge_map))

    out_path = PROJECT_ROOT / "results" / "tuan8" / "bang_so_sanh_3_cau_hinh_50cau.csv"
    write_summary_csv(summary_rows, out_path)
    print(f"\n[OK] Đã ghi bảng tổng hợp: {out_path}")
    print_config_reports(merged)


if __name__ == "__main__":
    main()

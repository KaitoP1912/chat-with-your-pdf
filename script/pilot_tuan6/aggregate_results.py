"""
script/pilot_tuan6/aggregate_results.py — Tuần 6, bước cuối: gộp 3 CSV kết quả
(page_aware, fixed_size, longcontext) thành 1 bảng so sánh duy nhất cho báo cáo.

LƯU Ý QUAN TRỌNG: answer_correctness (3 mức: đúng hoàn toàn / đúng một phần /
sai) KHÔNG thể tự động chấm — cột "answer_correctness_manual" trong mỗi CSV
phải được bạn tự điền tay (so answer_text với answer_reference trong
test_questions.json) TRƯỚC KHI chạy script này, nếu muốn có số liệu đó trong
bảng tổng hợp. Nếu chưa điền, script vẫn chạy bình thường và chỉ báo cáo các
chỉ số tự động được (Hit@3, citation accuracy, tỉ lệ abstain đúng/sai,
latency, token) — đây vẫn là phần lớn số liệu cần cho báo cáo.

Quy ước điền answer_correctness_manual (dùng đúng 3 giá trị này, không dấu,
để script đếm được):
    full      = đúng hoàn toàn
    partial   = đúng một phần
    wrong     = sai
(để trống nếu là dòng abstain/error, không cần chấm)

Cách dùng:
    python script/pilot_tuan6/aggregate_results.py \
        --page-aware results/tuan6_pilot/test_qa_results_page_aware.csv \
        --fixed-size results/tuan6_pilot/test_qa_results_fixed_size.csv \
        --longcontext results/tuan6_pilot/test_qa_results_longcontext.csv \
        --out results/tuan6_pilot/bang_so_sanh_3_cau_hinh.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def _to_float(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(rows: List[dict], config_label: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"config": config_label, "n": 0}

    answerable = [r for r in rows if _to_bool(r.get("is_answerable"))]
    unanswerable = [r for r in rows if not _to_bool(r.get("is_answerable"))]
    errors = [r for r in rows if _to_bool(r.get("is_error"))]
    non_error = [r for r in rows if not _to_bool(r.get("is_error"))]

    # Hit@3 chỉ áp dụng cho 2 cấu hình RAG (longcontext không có cột này -> rỗng, bỏ qua)
    hit3_rows = [r for r in answerable if str(r.get("hit_at_3", "")).strip() != ""]
    hit3_rate = (
        sum(1 for r in hit3_rows if _to_bool(r.get("hit_at_3"))) / len(hit3_rows)
        if hit3_rows else None
    )

    # Citation accuracy: chỉ tính trên câu answerable, không lỗi, không abstain
    citation_rows = [
        r for r in answerable
        if not _to_bool(r.get("is_error")) and str(r.get("citation_correct", "")).strip() != ""
    ]
    citation_rate = (
        sum(1 for r in citation_rows if _to_bool(r.get("citation_correct"))) / len(citation_rows)
        if citation_rows else None
    )

    # False acceptance: câu KHÔNG answerable nhưng hệ thống KHÔNG abstain (trả lời bừa)
    false_accept_rows = [r for r in unanswerable if not _to_bool(r.get("is_error"))]
    false_accept_rate = (
        sum(1 for r in false_accept_rows if not _to_bool(r.get("is_abstained"))) / len(false_accept_rows)
        if false_accept_rows else None
    )

    # False refusal: câu answerable nhưng hệ thống lại abstain (từ chối oan)
    false_refusal_rows = [r for r in answerable if not _to_bool(r.get("is_error"))]
    false_refusal_rate = (
        sum(1 for r in false_refusal_rows if _to_bool(r.get("is_abstained"))) / len(false_refusal_rows)
        if false_refusal_rows else None
    )

    # Answer correctness (3 mức), chỉ tính nếu cột đã được điền tay
    manual = [str(r.get("answer_correctness_manual", "")).strip().lower() for r in rows]
    graded = [m for m in manual if m in ("full", "partial", "wrong")]
    correctness = None
    if graded:
        correctness = {
            "n_graded": len(graded),
            "full_pct": graded.count("full") / len(graded),
            "partial_pct": graded.count("partial") / len(graded),
            "wrong_pct": graded.count("wrong") / len(graded),
        }

    latencies = [_to_float(r.get("latency_seconds")) for r in non_error]
    latencies = [x for x in latencies if x is not None]
    tokens = [_to_float(r.get("total_tokens")) for r in non_error]
    tokens = [x for x in tokens if x is not None]

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
        "answer_correctness": correctness,
        "avg_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
        "avg_total_tokens": (sum(tokens) / len(tokens)) if tokens else None,
    }


def _fmt_pct(x):
    return "" if x is None else f"{x * 100:.1f}%"


def _fmt_num(x, digits=2):
    return "" if x is None else f"{x:.{digits}f}"


def print_report(summaries: List[dict]) -> None:
    print("\n" + "=" * 78)
    print("BẢNG SO SÁNH 3 CẤU HÌNH — TEST SET")
    print("=" * 78)
    for s in summaries:
        if s["n"] == 0:
            print(f"\n[{s['config']}] — không có dữ liệu (file rỗng hoặc không tìm thấy).")
            continue
        print(f"\n[{s['config']}]  (n={s['n']}, answerable={s['n_answerable']}, "
              f"unanswerable={s['n_unanswerable']}, lỗi={s['n_errors']})")
        print(f"  Hit@3 (retriever)      : {_fmt_pct(s['hit_at_3_rate'])}"
              + ("" if s['hit_at_3_rate'] is not None else "  (n/a — cấu hình không có bước retrieval)"))
        print(f"  Citation accuracy      : {_fmt_pct(s['citation_accuracy'])}")
        print(f"  False acceptance rate  : {_fmt_pct(s['false_acceptance_rate'])}"
              " (câu KHÔNG có đáp án nhưng hệ thống vẫn trả lời — càng thấp càng tốt)")
        print(f"  False refusal rate     : {_fmt_pct(s['false_refusal_rate'])}"
              " (câu CÓ đáp án nhưng hệ thống từ chối oan — càng thấp càng tốt)")
        if s["answer_correctness"]:
            c = s["answer_correctness"]
            print(f"  Answer correctness     : full={_fmt_pct(c['full_pct'])} | "
                  f"partial={_fmt_pct(c['partial_pct'])} | wrong={_fmt_pct(c['wrong_pct'])} "
                  f"(đã chấm tay {c['n_graded']}/{s['n']} dòng)")
        else:
            print(f"  Answer correctness     : (chưa điền answer_correctness_manual)")
        print(f"  Latency trung bình     : {_fmt_num(s['avg_latency_seconds'])}s")
        print(f"  Token trung bình       : {_fmt_num(s['avg_total_tokens'], 0)}")
    print("\n" + "=" * 78)


def write_csv(summaries: List[dict], out_path: Path) -> None:
    fieldnames = [
        "config", "n", "n_answerable", "n_unanswerable", "n_errors",
        "hit_at_3_rate", "citation_accuracy", "false_acceptance_rate",
        "false_refusal_rate", "answer_full_pct", "answer_partial_pct",
        "answer_wrong_pct", "n_graded", "avg_latency_seconds", "avg_total_tokens",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            c = s.get("answer_correctness") or {}
            writer.writerow({
                "config": s["config"],
                "n": s.get("n", 0),
                "n_answerable": s.get("n_answerable", ""),
                "n_unanswerable": s.get("n_unanswerable", ""),
                "n_errors": s.get("n_errors", ""),
                "hit_at_3_rate": _fmt_pct(s.get("hit_at_3_rate")),
                "citation_accuracy": _fmt_pct(s.get("citation_accuracy")),
                "false_acceptance_rate": _fmt_pct(s.get("false_acceptance_rate")),
                "false_refusal_rate": _fmt_pct(s.get("false_refusal_rate")),
                "answer_full_pct": _fmt_pct(c.get("full_pct")),
                "answer_partial_pct": _fmt_pct(c.get("partial_pct")),
                "answer_wrong_pct": _fmt_pct(c.get("wrong_pct")),
                "n_graded": c.get("n_graded", ""),
                "avg_latency_seconds": _fmt_num(s.get("avg_latency_seconds")),
                "avg_total_tokens": _fmt_num(s.get("avg_total_tokens"), 0),
            })
    print(f"\nĐã ghi bảng tổng hợp vào: {out_path}")


def load_csv(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[CẢNH BÁO] Không tìm thấy file: {path} — bỏ qua cấu hình này.")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gộp 3 CSV kết quả test thành 1 bảng so sánh.")
    parser.add_argument("--page-aware", default="results/tuan6_pilot/test_qa_results_page_aware.csv")
    parser.add_argument("--fixed-size", default="results/tuan6_pilot/test_qa_results_fixed_size.csv")
    parser.add_argument("--longcontext", default="results/tuan6_pilot/test_qa_results_longcontext.csv")
    parser.add_argument("--out", default="results/tuan6_pilot/bang_so_sanh_3_cau_hinh.csv")
    args = parser.parse_args()

    summaries = [
        summarize(load_csv(args.page_aware), "page_aware"),
        summarize(load_csv(args.fixed_size), "fixed_size"),
        summarize(load_csv(args.longcontext), "longcontext"),
    ]

    print_report(summaries)
    write_csv(summaries, Path(args.out))


if __name__ == "__main__":
    main()
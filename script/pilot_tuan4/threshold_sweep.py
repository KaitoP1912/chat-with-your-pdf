"""
script/pilot_tuan4/threshold_sweep.py — Bước 2b của Tuần 4

Đọc CSV thô từ run_dev_retrieval.py (điểm top1_score mỗi câu x mỗi cấu hình),
quét dải ngưỡng tau, tính các chỉ số theo đúng 2 vai trò ngưỡng mà Thầy yêu
cầu (email 16/08/2026):
  (a) loại evidence quá yếu (đo qua false_abstention_rate trên câu answerable)
  (b) tín hiệu "không tìm thấy thông tin" (đo qua false_acceptance_rate và
      Abstention Precision/Recall/F1 trên câu unanswerable)

KHÔNG tự động khóa ngưỡng — script chỉ XUẤT BẢNG SỐ LIỆU + gợi ý theo 1 tiêu
chí mặc định (cân bằng false abstention/acceptance, ưu tiên nhẹ giảm false
acceptance vì rủi ro hallucination cao hơn). Người dùng đọc bảng, tự quyết
định và ghi lý do vào báo cáo — đúng yêu cầu "không chọn cảm tính" nhưng vẫn
là quyết định có ý thức của người làm, không phải máy tự chọn hộ.

Cách chạy:
    python script/pilot_tuan4/threshold_sweep.py \
        --input results/tuan4_pilot/dev_retrieval_raw.csv \
        --out-sweep results/tuan4_pilot/threshold_sweep.csv \
        --out-summary results/tuan4_pilot/threshold_recommendation.csv \
        --min 0.30 --max 0.80 --step 0.05
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["top1_score"] = float(r["top1_score"])
        r["is_answerable"] = r["is_answerable"].strip().lower() in ("true", "1", "yes")
    return rows


def frange(start: float, stop: float, step: float):
    n = round((stop - start) / step)
    for i in range(n + 1):
        yield round(start + i * step, 4)


def compute_metrics_at_threshold(rows: List[dict], tau: float) -> dict:
    """rows: chỉ các câu của 1 cấu hình (page_aware HOẶC fixed_size)."""
    answerable = [r for r in rows if r["is_answerable"]]
    unanswerable = [r for r in rows if not r["is_answerable"]]
    near_miss = [r for r in unanswerable if r.get("type") == "near_miss"]
    out_of_scope = [r for r in unanswerable if r.get("type") == "out_of_scope"]

    n_answerable = len(answerable)
    n_unanswerable = len(unanswerable)

    # "Trả lời" nếu top1_score >= tau, "từ chối" (abstain) nếu < tau
    false_abstention = sum(1 for r in answerable if r["top1_score"] < tau)
    false_acceptance = sum(1 for r in unanswerable if r["top1_score"] >= tau)
    false_acceptance_near_miss = sum(1 for r in near_miss if r["top1_score"] >= tau)
    false_acceptance_out_of_scope = sum(1 for r in out_of_scope if r["top1_score"] >= tau)

    false_abstention_rate = false_abstention / n_answerable if n_answerable else 0.0
    false_acceptance_rate = false_acceptance / n_unanswerable if n_unanswerable else 0.0
    near_miss_acceptance_rate = false_acceptance_near_miss / len(near_miss) if near_miss else 0.0
    out_of_scope_acceptance_rate = false_acceptance_out_of_scope / len(out_of_scope) if out_of_scope else 0.0

    # Abstention Precision/Recall/F1 (dự đoán "abstain" = top1_score < tau)
    predicted_abstain_correct = sum(1 for r in unanswerable if r["top1_score"] < tau)   # TP
    predicted_abstain_total = sum(1 for r in rows if r["top1_score"] < tau)              # TP + FP
    actual_unanswerable_total = n_unanswerable                                            # TP + FN

    abstention_recall = predicted_abstain_correct / actual_unanswerable_total if actual_unanswerable_total else 0.0
    abstention_precision = predicted_abstain_correct / predicted_abstain_total if predicted_abstain_total else 0.0
    if abstention_precision + abstention_recall > 0:
        abstention_f1 = 2 * abstention_precision * abstention_recall / (abstention_precision + abstention_recall)
    else:
        abstention_f1 = 0.0

    balanced_error_rate = (false_abstention_rate + false_acceptance_rate) / 2

    return {
        "threshold": tau,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "false_abstention_rate": round(false_abstention_rate, 4),
        "false_acceptance_rate": round(false_acceptance_rate, 4),
        "near_miss_acceptance_rate": round(near_miss_acceptance_rate, 4),
        "out_of_scope_acceptance_rate": round(out_of_scope_acceptance_rate, 4),
        "abstention_precision": round(abstention_precision, 4),
        "abstention_recall": round(abstention_recall, 4),
        "abstention_f1": round(abstention_f1, 4),
        "balanced_error_rate": round(balanced_error_rate, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Quét dải ngưỡng cosine, tính chỉ số abstention detection.")
    parser.add_argument("--input", default="results/tuan4_pilot/dev_retrieval_raw.csv",
                         help="CSV thô từ run_dev_retrieval.py (mặc định: results/tuan4_pilot/dev_retrieval_raw.csv)")
    parser.add_argument("--out-sweep", default="results/tuan4_pilot/threshold_sweep.csv",
                         help="CSV đầy đủ mọi ngưỡng x mọi cấu hình")
    parser.add_argument("--out-summary", default="results/tuan4_pilot/threshold_recommendation.csv",
                         help="CSV tóm tắt ngưỡng đề xuất theo từng cấu hình")
    parser.add_argument("--min", type=float, default=0.30)
    parser.add_argument("--max", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_rows(args.input)

    by_config: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    sweep_rows = []
    for config_name, config_rows in by_config.items():
        for tau in frange(args.min, args.max, args.step):
            metrics = compute_metrics_at_threshold(config_rows, tau)
            metrics["config"] = config_name
            sweep_rows.append(metrics)

    out_sweep_path = Path(args.out_sweep)
    out_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config"] + [k for k in sweep_rows[0].keys() if k != "config"]
    with open(out_sweep_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow(row)
    print(f"Đã ghi bảng sweep đầy đủ ({len(sweep_rows)} dòng) vào: {out_sweep_path}")

    # Gợi ý ngưỡng theo từng cấu hình: ưu tiên abstention_f1 cao nhất; nếu hòa,
    # chọn tau có false_acceptance_rate thấp hơn (an toàn hơn, giảm rủi ro
    # hallucination) — đây là lựa chọn mặc định, có thể đổi tiêu chí nếu Thầy
    # có ý kiến khác (xem câu hỏi mở ở cuối kế hoạch Tuần 4).
    summary_rows = []
    print("\n=== Gợi ý ngưỡng theo từng cấu hình (ưu tiên Abstention F1, tie-break: false_acceptance_rate thấp) ===")
    for config_name, config_rows in by_config.items():
        candidates = [r for r in sweep_rows if r["config"] == config_name]
        best = max(candidates, key=lambda r: (r["abstention_f1"], -r["false_acceptance_rate"]))
        summary_rows.append(best)
        print(f"\n[{config_name}]")
        print(f"  Ngưỡng đề xuất       : {best['threshold']}")
        print(f"  Abstention F1        : {best['abstention_f1']}")
        print(f"  False abstention rate: {best['false_abstention_rate']} "
              f"(tỉ lệ câu answerable bị từ chối oan)")
        print(f"  False acceptance rate: {best['false_acceptance_rate']} "
              f"(tỉ lệ câu unanswerable bị chấp nhận sai)")
        print(f"  - near_miss riêng    : {best['near_miss_acceptance_rate']}")
        print(f"  - out_of_scope riêng : {best['out_of_scope_acceptance_rate']}")

    out_summary_path = Path(args.out_summary)
    with open(out_summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["config"] + [k for k in summary_rows[0].keys() if k != "config"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\nĐã ghi bảng tóm tắt vào: {out_summary_path}")
    print("\nLưu ý: đây là GỢI Ý theo 1 tiêu chí mặc định, không phải quyết định cuối."
          "\nMở threshold_sweep.csv, nhìn cả dải ngưỡng lân cận để tự quyết định,"
          "\nghi rõ lý do chọn vào báo cáo Tuần 4 (đúng yêu cầu của Thầy: không chọn cảm tính).")


if __name__ == "__main__":
    main()
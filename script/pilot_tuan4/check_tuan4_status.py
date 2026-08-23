"""
script/pilot_tuan4/check_tuan4_status.py

Kiểm tra tiến độ Tuần 4 bằng cách soi các file kết quả đã có, đối chiếu với
7 bước trong kế hoạch. KHÔNG chạy lại retrieval/threshold, chỉ đọc file có sẵn.

Cách chạy (từ thư mục gốc dự án):
    python script/pilot_tuan4/check_tuan4_status.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "[DONE]" if ok else "[CHƯA]"
    line = f"  {mark} {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def main():
    print("="*70)
    print("KIỂM TRA TIẾN ĐỘ TUẦN 4")
    print("="*70)

    # --- Bước 1: Dev set mở rộng ---
    print("\nBước 1 — Dev set mở rộng (answerable + unanswerable)")
    dev_path = PROJECT_ROOT / "data/eval_sets/dev_questions_normalized.json"
    if dev_path.exists():
        with open(dev_path, encoding="utf-8") as f:
            data = json.load(f)
        qs = data["questions"]
        n_ans = sum(1 for q in qs if q["is_answerable"])
        n_unans = len(qs) - n_ans
        n_near = sum(1 for q in qs if q.get("type") == "near_miss")
        n_oos = sum(1 for q in qs if q.get("type") == "out_of_scope")
        check("File dev_questions_normalized.json tồn tại", True)
        check(f"Tổng {len(qs)} câu ({n_ans} answerable / {n_unans} unanswerable: "
              f"{n_oos} out_of_scope + {n_near} near_miss)", n_unans >= 8)
    else:
        check("File dev_questions_normalized.json tồn tại", False,
              str(dev_path))

    # --- Bước 2: Threshold calibration ---
    print("\nBước 2 — Threshold calibration")
    raw_path = PROJECT_ROOT / "results/tuan4_pilot/dev_retrieval_raw.csv"
    sweep_path = PROJECT_ROOT / "results/tuan4_pilot/threshold_sweep.csv"
    reco_path = PROJECT_ROOT / "results/tuan4_pilot/threshold_recommendation.csv"

    hit_rates = {}
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        check(f"dev_retrieval_raw.csv tồn tại ({len(rows)} dòng)", True)
        for config in {r["config"] for r in rows}:
            sub = [r for r in rows if r["config"] == config and r["is_answerable"].lower() == "true"]
            hits = sum(1 for r in sub if r.get("hit_at_3", "").lower() == "true")
            rate = hits / len(sub) if sub else 0
            hit_rates[config] = rate
            print(f"       Hit@3 [{config}]: {hits}/{len(sub)} = {rate*100:.1f}%")
    else:
        check("dev_retrieval_raw.csv tồn tại", False, str(raw_path))

    check("threshold_sweep.csv tồn tại", sweep_path.exists(), str(sweep_path))
    check("threshold_recommendation.csv tồn tại", reco_path.exists(), str(reco_path))

    if reco_path.exists():
        with open(reco_path, encoding="utf-8") as f:
            reco_rows = list(csv.DictReader(f))
        for r in reco_rows:
            print(f"       Ngưỡng đề xuất [{r['config']}]: tau={r['threshold']}, "
                  f"Abstention F1={r['abstention_f1']}")

    # --- Bước 3: Metric abstention trong rubric ---
    print("\nBước 3 — Bổ sung Abstention Precision/Recall/F1 vào rubric (văn bản)")
    check("Đã có số liệu (từ threshold_recommendation.csv)", reco_path.exists(),
          "còn thiếu: cập nhật rubric.md / báo cáo bằng văn bản")

    # --- Bước 4: Tích hợp Gemini QA ---
    print("\nBước 4 — Tích hợp Gemini QA & Citation Generation")
    qa_candidates = list((PROJECT_ROOT / "source/qa").glob("*.py")) if (PROJECT_ROOT / "source/qa").exists() else []
    check("Thư mục source/qa/ có module QA", len(qa_candidates) > 0,
          f"tìm thấy: {[p.name for p in qa_candidates]}" if qa_candidates else "CHƯA LÀM — bước tiếp theo")

   # --- Bước 5: Chạy song song 2 cấu hình full metric ---
    print("\nBước 5 — Đo đầy đủ Hit@3 + answer correctness + citation accuracy + abstention + latency + token")
    check("Đã có Hit@3 (từ dev_retrieval_raw.csv)", raw_path.exists())
    
    qa_path = PROJECT_ROOT / "results/tuan4_pilot/dev_qa_results.csv"
    if qa_path.exists():
        with open(qa_path, encoding="utf-8") as f:
            qa_rows = list(csv.DictReader(f))
        n_done = sum(1 for r in qa_rows if r.get("is_error", "").lower() != "true")
        check(f"Đã chạy Gemini QA ({n_done}/{len(qa_rows)} dòng thành công)", n_done == len(qa_rows) and len(qa_rows) > 0)
    else:
        check("dev_qa_results.csv tồn tại", False, "Cần chạy run_dev_qa.py trước")

    # --- Bước 6: Áp quy tắc quyết định ---
    print("\nBước 6 — Áp quy tắc '≤3pp Hit@3' để quyết định khóa page-aware")
    if hit_rates.get("page_aware") is not None and hit_rates.get("fixed_size") is not None:
        diff_pp = (hit_rates["page_aware"] - hit_rates["fixed_size"]) * 100
        if hit_rates["page_aware"] >= hit_rates["fixed_size"]:
            verdict = f"page_aware CAO HƠN fixed_size {diff_pp:.1f}pp -> ĐƯỢC PHÉP khóa page-aware ngay"
        elif diff_pp >= -3:
            verdict = f"page_aware thấp hơn {abs(diff_pp):.1f}pp (trong ngưỡng 3pp) -> vẫn khóa page-aware"
        else:
            verdict = f"page_aware thấp hơn {abs(diff_pp):.1f}pp (VƯỢT 3pp) -> phải phân tích lỗi trước khi khóa"
        check("Đã có đủ số liệu để áp quy tắc", True, verdict)
    else:
        check("Đã có đủ số liệu để áp quy tắc", False, "cần Bước 2 xong trước")

    # --- Bước 7: Nghiệm thu ---
    print("\nBước 7 — Nghiệm thu Tuần 4 (viết báo cáo)")
    check("Đủ số liệu để viết báo cáo Tuần 4", raw_path.exists() and reco_path.exists(),
          "còn thiếu phần Gemini QA (Bước 4-5) nếu muốn đủ 'thành phần cốt lõi' end-to-end")

    print("\n" + "="*70)
    print("TÓM TẮT: đã xong Bước 1, 2 (và có thể chốt luôn Bước 6 nếu Hit@3 page_aware >= fixed_size).")
    print("Việc tiếp theo cần làm: Bước 4 — tích hợp Gemini sinh câu trả lời kèm trích dẫn trang.")
    print("="*70)


if __name__ == "__main__":
    main()
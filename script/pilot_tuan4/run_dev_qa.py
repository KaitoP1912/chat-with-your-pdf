"""
script/pilot_tuan4/run_dev_qa.py — Bước 4.3 + 5 của Tuần 4

Chạy Gemini QA generation (qa_generator.generate_answer) trên dev set cho cả
2 cấu hình chunking, đo latency/token, tự động kiểm citation accuracy (so
citation trả về với expected_page), xuất CSV cho báo cáo Bước 5.

HỖ TRỢ 2 KỊCH BẢN NGƯỠNG TAU SONG SONG (để trình bày cho Thầy so sánh, xem
báo cáo Tuần 4 mục 3.4):

  - Kịch bản 1 "Two-Tier Abstention": DÙNG CHUNG 1 ngưỡng tau=0.38 cho cả 2
    cấu hình. Ngưỡng này được suy ra từ khoảng trống giữa điểm cosine thấp
    nhất của câu answerable (dev_13=0.3994) và điểm cao nhất của 1 câu
    unanswerable cụ thể (dev_32=0.3723) trong dev_retrieval_raw.csv — KHÔNG
    phải từ threshold_sweep.py. Mục đích: giữ đúng 1 biến số duy nhất
    (cách chunking) khi so sánh 2 cấu hình, đổi lại chấp nhận 1 số câu
    unanswerable lọt qua tầng Retrieval, phải trông cậy vào tầng Model
    Refusal (Gemini tự chối) chặn tiếp ở sau — CHƯA có gì đảm bảo tầng 2 sẽ
    luôn chặn đúng 100% cho các câu hỏi khác trong tương lai.
    Chạy: --tau 0.38

  - Kịch bản 2 "Strict Sweep Filter" (mặc định của script): DÙNG RIÊNG
    ngưỡng theo đúng threshold_sweep.py — page_aware=0.5, fixed_size=0.45.
    Ngưỡng "production" thực sự nên là ngưỡng riêng của page_aware (kiến
    trúc mặc định của ứng dụng đã khóa), chặn thẳng ở tầng Retrieval, không
    cần tin tưởng hành vi tự chối của model.
    Chạy: (không cần cờ, đây là mặc định) hoặc --tau-page-aware 0.5
    --tau-fixed-size 0.45

Cách dùng --tau: nếu truyền --tau, giá trị này ÁP DỤNG CHUNG cho cả 2 cấu
hình, ghi đè --tau-page-aware/--tau-fixed-size (nếu có truyền cùng lúc).

answer_correctness (3 mức: đúng hoàn toàn/đúng một phần/sai) KHÔNG tự động
chấm được — để cột trống, người dùng đọc answer_text + answer_reference rồi
tự điền tay.

Cách chạy Kịch bản 1 (tau=0.38 chung, ghi ra file riêng để không đè lên
Kịch bản 2):
    python script/pilot_tuan4/run_dev_qa.py --limit 0 --sleep 5.0 --tau 0.38 \
        --out results/tuan4_pilot/dev_qa_results_scenario1_tau038.csv

Cách chạy Kịch bản 2 (mặc định, tau riêng theo sweep):
    python script/pilot_tuan4/run_dev_qa.py --limit 0 --sleep 5.0 \
        --out results/tuan4_pilot/dev_qa_results_scenario2_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from source.retrieval.vectorstore import search, ChunkIndex
from source.qa.qa_generator import generate_answer, DEFAULT_MODEL
from run_dev_retrieval import CONFIGS, build_indices_for_file, is_hit

# Kịch bản 2 (mặc định) — ngưỡng riêng theo threshold_recommendation.csv.
# page_aware là kiến trúc mặc định đã được duyệt -> ngưỡng 0.5 là ngưỡng
# "production" thực sự. fixed_size=0.45 chỉ dùng để chạy bảng so sánh.
TAU_PAGE_AWARE_DEFAULT = 0.5
TAU_FIXED_SIZE_DEFAULT = 0.45


def main():
    parser = argparse.ArgumentParser(description="Chạy Gemini QA trên dev set, đo latency/token/citation.")
    parser.add_argument("--dev-set", default="data/eval_sets/dev_questions_normalized.json")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--vncorenlp-dir", default="vncorenlp_models")
    parser.add_argument("--k", type=int, default=3, help="Top-k đưa vào QA, khớp k đã dùng ở Bước 2")
    parser.add_argument("--tau", type=float, default=None,
                        help="[Kịch bản 1] Ngưỡng tau DÙNG CHUNG cho cả 2 cấu hình. "
                             "Nếu truyền, ghi đè --tau-page-aware/--tau-fixed-size.")
    parser.add_argument("--tau-page-aware", type=float, default=TAU_PAGE_AWARE_DEFAULT,
                        help=f"[Kịch bản 2, mặc định] Ngưỡng riêng cho page_aware (mặc định {TAU_PAGE_AWARE_DEFAULT})")
    parser.add_argument("--tau-fixed-size", type=float, default=TAU_FIXED_SIZE_DEFAULT,
                        help=f"[Kịch bản 2, mặc định] Ngưỡng riêng cho fixed_size (mặc định {TAU_FIXED_SIZE_DEFAULT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chỉ định model cố định cho phiên chạy")
    parser.add_argument("--out", default="results/tuan4_pilot/dev_qa_results.csv",
                        help="QUAN TRỌNG: đặt tên khác nhau cho mỗi kịch bản để không ghi đè lẫn nhau, "
                             "vd dev_qa_results_scenario1_tau038.csv vs dev_qa_results_scenario2_sweep.csv")
    parser.add_argument("--limit", type=int, default=3,
                        help="Chỉ chạy N câu đầu tiên để test (mặc định 3, dry-run). "
                             "Dùng --limit 0 để chạy FULL toàn bộ dev set.")
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="Số giây nghỉ giữa 2 lần gọi Gemini (mặc định 5s cho model flash-lite 15 RPM)")
    parser.add_argument("--resume", action="store_true",
                        help="Đọc lại --out cũ (nếu có), bỏ qua các câu ĐÃ thành công. "
                             "CẢNH BÁO: không dùng cờ này khi chuyển giữa 2 kịch bản khác tau, vì các "
                             "dòng cũ sẽ bị giữ nguyên theo tau cũ, không được tính lại.")
    args = parser.parse_args()

    if args.tau is not None:
        tau_by_config = {"page_aware": args.tau, "fixed_size": args.tau}
        scenario_label = f"KỊCH BẢN 1 (Two-Tier Abstention) — tau chung = {args.tau}"
    else:
        tau_by_config = {"page_aware": args.tau_page_aware, "fixed_size": args.tau_fixed_size}
        scenario_label = (f"KỊCH BẢN 2 (Strict Sweep Filter) — page_aware={args.tau_page_aware}, "
                           f"fixed_size={args.tau_fixed_size}")

    print(f"=== {scenario_label} ===")
    print(f"Output: {args.out}\n")

    corpus_dir = Path(args.corpus_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vncorenlp_dir = str(Path(args.vncorenlp_dir).resolve())

    with open(Path(args.dev_set).resolve(), "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    questions = dev_data["questions"]

    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
        print(f"*** DRY-RUN: chỉ chạy {len(questions)} câu đầu tiên. "
              f"Dùng --limit 0 để chạy full sau khi kiểm tra ổn. ***\n")

    # --- Resume: đọc kết quả cũ (nếu có), lấy ra các câu ĐÃ thành công để khỏi gọi lại ---
    previous_ok: Dict[tuple, dict] = {}
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        for r in old_rows:
            if r.get("is_error", "").lower() != "true":
                previous_ok[(r["id"], r["config"])] = r
        print(f"*** RESUME: tìm thấy {len(previous_ok)} dòng đã thành công ở lần chạy trước, "
              f"sẽ bỏ qua, chỉ gọi lại phần còn thiếu/lỗi. ***\n")

    unique_files = sorted({q["source_file"] for q in questions})
    print(f"Build index cho {len(unique_files)} file: {unique_files}\n")

    all_indices: Dict[str, Dict[str, ChunkIndex]] = {}
    for source_file in unique_files:
        print(f"[Build index] {source_file}")
        all_indices[source_file] = build_indices_for_file(source_file, corpus_dir, vncorenlp_dir)
        print()

    rows = []
    total = len(questions) * len(CONFIGS)
    done = 0

    for q in questions:
        source_file = q["source_file"]
        expected_pages = [int(p) for p in (q.get("expected_page") or [])] if q.get("expected_page") else []
        indices = all_indices[source_file]

        for config_name in CONFIGS:
            done += 1

            key = (q["id"], config_name)
            if key in previous_ok:
                rows.append(previous_ok[key])
                print(f"  [{done}/{total}] {q['id']:8s} ({config_name:10s}) -> REUSED (đã thành công trước đó)")
                continue

            index = indices[config_name]
            hits = search(index, q["question"], vncorenlp_dir, k=args.k)

            tau = tau_by_config[config_name]
            answer = generate_answer(q["question"], hits, tau=tau, target_model=args.model)

            citation_correct = ""
            if q["is_answerable"] and not answer.is_abstained and not answer.is_error:
                any_correct = any(
                    is_hit(c.get("page_number"), c.get("page_range"), expected_pages)
                    for c in answer.citations
                )
                citation_correct = any_correct

            rows.append({
                "id": q["id"],
                "config": config_name,
                "tau_used": tau,
                "is_answerable": q["is_answerable"],
                "type": q.get("type", ""),
                "question": q["question"],
                "expected_page": ";".join(str(p) for p in expected_pages),
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
                "model_used": answer.model_used or "",
                "answer_correctness_manual": "",
            })

            status = "ABSTAIN" if answer.is_abstained else ("ERROR" if answer.is_error else "OK")
            print(f"  [{done}/{total}] {q['id']:8s} ({config_name:10s}, tau={tau}) -> {status} "
                  f"latency={answer.latency_seconds} tokens={answer.total_tokens}")

            if answer.abstain_reason != "retrieval_threshold":
                time.sleep(args.sleep)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(1 for r in rows if str(r["is_error"]).strip().lower() == "true")
    print(f"\nĐã ghi {len(rows)} dòng vào: {out_path}")
    if n_errors:
        print(f"[CẢNH BÁO] {n_errors} dòng bị lỗi API.")

    models_used = {r["model_used"] for r in rows
                   if str(r["is_abstained"]).strip().lower() != "true" and r.get("model_used")}
    if len(models_used) > 1:
        print(f"\n[CẢNH BÁO QUAN TRỌNG] Phát hiện {len(models_used)} model KHÁC NHAU: {sorted(models_used)}")
    elif models_used:
        print(f"\n[OK] Toàn bộ các câu có gọi API đều dùng cùng 1 model: {list(models_used)[0]}")

    slow_rows = [r for r in rows if r.get("latency_seconds") not in ("", None) and float(r["latency_seconds"]) > 30]
    if slow_rows:
        print(f"\n[LƯU Ý] {len(slow_rows)} dòng có latency > 30s (nghi retry rate-limit):")
        for r in slow_rows:
            print(f"   - {r['id']} ({r['config']}): {r['latency_seconds']}s")

    # Đối chiếu nhanh riêng cho các câu unanswerable: kiểm tra có bao nhiêu câu
    # lọt qua tầng retrieval_threshold (tức phải nhờ tầng model_refusal chặn tiếp)
    unans_rows = [r for r in rows if str(r["is_answerable"]).strip().lower() == "false"]
    leaked_to_model = [r for r in unans_rows if r["abstain_reason"] != "retrieval_threshold"]
    if leaked_to_model:
        print(f"\n[THỐNG KÊ KỊCH BẢN] {len(leaked_to_model)}/{len(unans_rows)} câu unanswerable "
              f"KHÔNG bị chặn ở tầng Retrieval (đã lọt qua tau={tau_by_config}, "
              f"phải nhờ tầng Model Refusal xử lý tiếp):")
        for r in leaked_to_model:
            final_status = "chặn đúng (model_refusal)" if r["is_abstained"] == "True" else "*** LỌT HẲN, TRẢ LỜI SAI ***"
            print(f"   - {r['id']}: {final_status}")

    print("\nBước tiếp theo: mở CSV, điền cột answer_correctness_manual cho các dòng có answer_text.")


if __name__ == "__main__":
    main()
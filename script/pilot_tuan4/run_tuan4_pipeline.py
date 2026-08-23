"""
script/pilot_tuan4/run_tuan4_pipeline.py — chạy gộp Bước 2a + 2b của Tuần 4

Chạy liên tiếp run_dev_retrieval.py rồi threshold_sweep.py bằng 1 lệnh duy
nhất, dùng toàn bộ giá trị mặc định đã khớp đúng cấu trúc dự án
(chat-with-your-pdf). Nếu 1 trong 2 bước lỗi, dừng ngay, không chạy tiếp
bước sau với dữ liệu rỗng.

Cách chạy (không cần tham số gì, chạy từ thư mục gốc dự án):
    python script/pilot_tuan4/run_tuan4_pipeline.py

Nếu cần đổi tham số (ví dụ đổi dải ngưỡng), truyền y hệt cú pháp của
run_dev_retrieval.py / threshold_sweep.py, script này sẽ tự chuyển tiếp:
    python script/pilot_tuan4/run_tuan4_pipeline.py --k 5 --min 0.25 --max 0.75
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script_name: str, args: list) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)] + args
    print(f"\n{'='*70}\n>>> Đang chạy: {' '.join(cmd)}\n{'='*70}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[DỪNG] {script_name} kết thúc với lỗi (mã {result.returncode}). "
              f"Không chạy tiếp bước sau. Xem log lỗi phía trên để sửa.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Chạy gộp retrieval + threshold sweep cho Tuần 4 bằng 1 lệnh."
    )
    # Tham số cho bước 1 (run_dev_retrieval.py)
    parser.add_argument("--dev-set", default="data/eval_sets/dev_questions_normalized.json")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--vncorenlp-dir", default="vncorenlp_models")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--raw-out", default="results/tuan4_pilot/dev_retrieval_raw.csv",
                         help="CSV thô output của bước retrieval (input của bước sweep)")
    # Tham số cho bước 2 (threshold_sweep.py)
    parser.add_argument("--out-sweep", default="results/tuan4_pilot/threshold_sweep.csv")
    parser.add_argument("--out-summary", default="results/tuan4_pilot/threshold_recommendation.csv")
    parser.add_argument("--min", type=float, default=0.30)
    parser.add_argument("--max", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    # --- Bước 1: retrieval ---
    run_step("run_dev_retrieval.py", [
        "--dev-set", args.dev_set,
        "--corpus-dir", args.corpus_dir,
        "--vncorenlp-dir", args.vncorenlp_dir,
        "--k", str(args.k),
        "--out", args.raw_out,
    ])

    # --- Bước 2: threshold sweep ---
    run_step("threshold_sweep.py", [
        "--input", args.raw_out,
        "--out-sweep", args.out_sweep,
        "--out-summary", args.out_summary,
        "--min", str(args.min),
        "--max", str(args.max),
        "--step", str(args.step),
    ])

    print(f"\n{'='*70}\nHOÀN TẤT cả 2 bước. Xem kết quả tại:\n"
          f"  - {args.raw_out}\n  - {args.out_sweep}\n  - {args.out_summary}\n{'='*70}")


if __name__ == "__main__":
    main()
"""
script/pilot_tuan4/run_dev_retrieval.py — Bước 2a của Tuần 4

Chạy retrieval trên dev set (dev_questions_normalized.json) với CẢ 2 cấu hình
chunking (page_aware = chunk_by_page, fixed_size = chunk_fixed_size), ghi lại
điểm cosine top-k cho từng câu.

Script này CHỈ ĐO, không tự chọn ngưỡng — output là input thô cho
threshold_sweep.py chạy ở bước sau.

Vì kiến trúc ứng dụng là "1 tài liệu / phiên" (đã chốt trong đề cương),
retrieval chạy RIÊNG trong phạm vi từng file nguồn của câu hỏi — build 2 FAISS
index (page_aware, fixed_size) cho MỖI file, không trộn cả 3 file vào 1 index
chung.

Cách chạy (ví dụ, sửa lại đường dẫn vncorenlp-dir cho đúng máy bạn):
    python script/pilot_tuan4/run_dev_retrieval.py ^
        --dev-set data/eval_sets/dev_questions_normalized.json ^
        --corpus-dir data/corpus ^
        --vncorenlp-dir D:/PHUOC/HK9/TTTN/chat-with-your-pdf/models/wordsegmenter ^
        --k 3 ^
        --out results/tuan4_pilot/dev_retrieval_raw.csv

Lưu ý: --vncorenlp-dir PHẢI là đường dẫn tuyệt đối (đúng ràng buộc đã ghi
trong word_segmenter.py ở Tuần 3 — đường dẫn tương đối gây lỗi JVM khó hiểu).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

# Cho phép chạy trực tiếp bằng `python script/pilot_tuan4/run_dev_retrieval.py`
# từ thư mục gốc dự án mà không cần set PYTHONPATH tay.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.chunker import chunk_by_page, chunk_fixed_size
from source.retrieval.vectorstore import build_index, search, ChunkIndex


CONFIGS = {
    "page_aware": chunk_by_page,
    "fixed_size": chunk_fixed_size,
}


def parse_page_range(page_range: str) -> List[int]:
    """'3-4' -> [3, 4]. Trả về [] nếu None/rỗng/không parse được."""
    if not page_range:
        return []
    try:
        return [int(p) for p in page_range.split("-")]
    except ValueError:
        return []


def is_hit(hit_page_number, hit_page_range, expected_pages: List[int]) -> bool:
    """1 hit được tính là đúng nếu page_number, hoặc bất kỳ trang nào trong
    page_range của bridge chunk, trùng với bất kỳ trang nào trong expected_pages.
    """
    if not expected_pages:
        return False
    if hit_page_number is not None and hit_page_number in expected_pages:
        return True
    if hit_page_range:
        for p in parse_page_range(hit_page_range):
            if p in expected_pages:
                return True
    return False


def build_indices_for_file(
    source_file: str, corpus_dir: Path, vncorenlp_dir: str
) -> Dict[str, ChunkIndex]:
    """Build 2 FAISS index (page_aware, fixed_size) cho 1 file PDF."""
    file_path = corpus_dir / source_file
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file corpus: {file_path}\n"
            f"Kiểm tra lại tên file trong dev set (source_file) có khớp đúng "
            f"tên file thật trong {corpus_dir} không (kể cả hoa/thường, "
            f"dấu gạch dưới, đuôi 33tr/53tr/60tr)."
        )

    print(f"  -> Nạp & chuẩn hóa (Trạm 1): {source_file}")
    pages = build_clean_pages(str(file_path))

    indices: Dict[str, ChunkIndex] = {}
    for config_name, chunk_fn in CONFIGS.items():
        print(f"     Chunking [{config_name}] ...")
        chunks = chunk_fn(pages)
        print(f"     -> {len(chunks)} chunk, đang embed + build FAISS ...")
        indices[config_name] = build_index(chunks, vncorenlp_dir)

    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Chạy retrieval dev set cho 2 cấu hình chunking, xuất CSV điểm cosine thô."
    )
    parser.add_argument("--dev-set", default="data/eval_sets/dev_questions_normalized.json",
                         help="Đường dẫn dev_questions_normalized.json (mặc định: data/eval_sets/dev_questions_normalized.json)")
    parser.add_argument("--corpus-dir", default="data/corpus",
                         help="Thư mục chứa PDF corpus (mặc định: data/corpus)")
    parser.add_argument("--vncorenlp-dir", default="vncorenlp_models",
                         help="Thư mục model VnCoreNLP, tương đối hoặc tuyệt đối đều được — "
                              "script tự resolve thành tuyệt đối (mặc định: vncorenlp_models)")
    parser.add_argument("--k", type=int, default=3, help="Số kết quả top-k lấy về mỗi câu (mặc định 3)")
    parser.add_argument("--out", default="results/tuan4_pilot/dev_retrieval_raw.csv",
                         help="Đường dẫn CSV output (mặc định: results/tuan4_pilot/dev_retrieval_raw.csv)")
    args = parser.parse_args()

    # Chuyển toàn bộ các đường dẫn tương đối thành tuyệt đối để tránh lỗi CWD bị thay đổi bởi py_vncorenlp/JVM
    corpus_dir = Path(args.corpus_dir).resolve()
    out_path = Path(args.out).resolve()
    dev_set_path = Path(args.dev_set).resolve()
    vncorenlp_dir = str(Path(args.vncorenlp_dir).resolve())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.vncorenlp_dir = vncorenlp_dir

    print(f"dev-set (đã resolve tuyệt đối): {dev_set_path}")
    print(f"corpus-dir (đã resolve tuyệt đối): {corpus_dir}")
    print(f"vncorenlp-dir (đã resolve tuyệt đối): {vncorenlp_dir}")

    with open(dev_set_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    questions = dev_data["questions"]

    # Build index 1 lần cho mỗi file nguồn, tái sử dụng cho mọi câu hỏi cùng file
    # (tránh embed lại toàn bộ corpus nhiều lần -> tiết kiệm thời gian đáng kể).
    unique_files = sorted({q["source_file"] for q in questions})
    print(f"Tổng số file nguồn cần build index: {len(unique_files)}")
    print(f"Danh sách: {unique_files}\n")

    all_indices: Dict[str, Dict[str, ChunkIndex]] = {}
    for source_file in unique_files:
        print(f"[Build index] {source_file}")
        all_indices[source_file] = build_indices_for_file(source_file, corpus_dir, args.vncorenlp_dir)
        print()

    # Chạy retrieval cho từng câu, từng cấu hình
    rows = []
    total = len(questions) * len(CONFIGS)
    done = 0

    hit_col = f"hit_at_{args.k}"

    for q in questions:
        source_file = q["source_file"]
        expected_pages = [int(p) for p in (q.get("expected_page") or [])] if q.get("expected_page") else []
        indices = all_indices[source_file]

        for config_name in CONFIGS:
            done += 1
            index = indices[config_name]
            hits = search(index, q["question"], args.vncorenlp_dir, k=args.k)

            top1_score = hits[0].score if hits else 0.0
            top1_page = hits[0].page_number if hits else None
            top1_page_range = hits[0].page_range if hits else None
            top1_is_bridge = hits[0].is_bridge if hits else False

            hit_at_k = any(is_hit(h.page_number, h.page_range, expected_pages) for h in hits)
            all_scores = ";".join(f"{h.score:.4f}" for h in hits)

            rows.append({
                "id": q["id"],
                "source_file": source_file,
                "config": config_name,
                "is_answerable": q["is_answerable"],
                "type": q.get("type", ""),
                "is_bridge_case": q.get("is_bridge_case", False),
                "expected_page": ";".join(str(p) for p in expected_pages),
                "num_hits": len(hits),
                "top1_score": round(top1_score, 4),
                "top1_page": top1_page if top1_page is not None else "",
                "top1_page_range": top1_page_range or "",
                "top1_is_bridge": top1_is_bridge,
                hit_col: hit_at_k,
                "all_topk_scores": all_scores,
            })

            print(f"  [{done}/{total}] {q['id']:8s} ({config_name:10s}) -> "
                  f"top1_score={top1_score:.4f}  {hit_col}={hit_at_k}")

    if not rows:
        print("Không có câu hỏi nào trong dev set — kiểm tra lại file --dev-set.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nĐã ghi {len(rows)} dòng ({len(questions)} câu x {len(CONFIGS)} cấu hình) vào: {out_path}")
    print("Bước tiếp theo: chạy threshold_sweep.py trên file CSV này.")


if __name__ == "__main__":
    main()
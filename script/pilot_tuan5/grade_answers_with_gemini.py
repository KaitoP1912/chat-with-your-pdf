"""
script/pilot_tuan5/grade_answers_with_gemini.py

Chấm cột answer_correctness_manual trong CSV kết quả QA bằng Gemini, dùng
CHÍNH nguyên văn trang PDF (expected_page) làm bằng chứng gốc — thay cho việc
gửi cả file PDF nặng qua chat để chấm tay.

CÁCH HOẠT ĐỘNG (mỗi câu answerable, 1 lần gọi Gemini):
  1. Trích text nguyên văn (các) trang expected_page bằng pdfplumber (đọc
     trực tiếp từ file PDF gốc trong corpus-dir, KHÔNG qua text_normalizer/
     chunker — dùng để đối chiếu độc lập, giống tinh thần oracle_context_test.py).
  2. Gửi cho Gemini: câu hỏi + nguyên văn trang + answer_text hệ thống đã trả
     lời, yêu cầu chấm đúng 1 trong 3 nhãn: "đúng hoàn toàn" / "đúng một phần"
     / "sai", kèm 1 câu lý do ngắn.
  3. Ghi nhãn vào cột answer_correctness_manual, lý do vào cột
     answer_correctness_reason (cột audit mới, không có trong bản gốc) để bạn
     kiểm tra nhanh chỗ nào Gemini chấm có thể sai mà không cần đọc lại PDF.

LƯU Ý QUAN TRỌNG — đây là chấm TỰ ĐỘNG bằng model, không thay thế hoàn toàn
việc bạn tự đọc: dùng để tiết kiệm thời gian chấm 23 câu, nhưng nên đọc lướt
qua answer_correctness_reason và soát lại NHỮNG CÂU chấm "sai" hoặc "đúng một
phần" bằng mắt trước khi đưa vào báo cáo chính thức nộp thầy — đúng nguyên
tắc "xác nhận bằng bằng chứng gốc" đã thống nhất, tự động hoá không miễn trừ
việc soát cuối.

Cách chạy:
    python script/pilot_tuan5/grade_answers_with_gemini.py \
        --results results/tuan5_pilot/dev_qa_results_v2_page_aware.csv \
        --dev-set data/eval_sets/dev_questions_normalized.json \
        --corpus-dir data/corpus \
        --out results/tuan5_pilot/dev_qa_results_v2_page_aware_graded.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
import google.generativeai as genai

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

GRADING_MODEL = "gemini-3.5-flash-lite"
GRADING_TEMPERATURE = 0.0
VALID_LABELS = ["đúng hoàn toàn", "đúng một phần", "sai"]


def ensure_configured() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "dán_key" in api_key:
        raise RuntimeError("Không tìm thấy GEMINI_API_KEY hợp lệ trong file .env.")
    genai.configure(api_key=api_key)


def extract_reference_pages(corpus_dir: str, source_file: str, pages: List[int]) -> str:
    """Trích nguyên văn (các) trang expected_page trực tiếp bằng pdfplumber,
    độc lập với pipeline chunking/normalize — dùng làm bằng chứng gốc để chấm."""
    if pdfplumber is None:
        raise RuntimeError("Thiếu thư viện pdfplumber (pip install pdfplumber).")

    file_path = os.path.join(corpus_dir, source_file)
    parts = []
    with pdfplumber.open(file_path) as pdf:
        for p in pages:
            if 1 <= p <= len(pdf.pages):
                text = pdf.pages[p - 1].extract_text(layout=False) or ""
                parts.append(f"[Trang {p}]\n{text.strip()}")
            else:
                parts.append(f"[Trang {p}] — NGOÀI PHẠM VI FILE, không trích được.")
    return "\n\n".join(parts)


def build_grading_prompt(question: str, reference_text: str, answer_text: str) -> str:
    labels_str = " / ".join(f'"{l}"' for l in VALID_LABELS)
    return (
        "Bạn là người chấm điểm khách quan cho hệ thống hỏi đáp tài liệu RAG.\n"
        "Nhiệm vụ: So sánh CÂU TRẢ LỜI CỦA HỆ THỐNG với NGUYÊN VĂN TRANG TÀI LIỆU "
        "(bằng chứng gốc, coi là đúng tuyệt đối), rồi chấm câu trả lời theo đúng "
        f"1 trong 3 nhãn: {labels_str}.\n\n"
        "Quy tắc chấm:\n"
        "- \"đúng hoàn toàn\": mọi thông tin, số liệu trong câu trả lời đều khớp với "
        "nguyên văn, không thiếu ý chính, không có chi tiết sai hoặc bịa thêm.\n"
        "- \"đúng một phần\": có it nhất 1 ý đúng, nhưng thiếu ý quan trọng hoặc có "
        "chi tiết/số liệu sai lệch so với nguyên văn.\n"
        "- \"sai\": thông tin chính trong câu trả lời không khớp hoặc không có "
        "trong nguyên văn (kể cả khi câu trả lời có tự nhận không chắc chắn).\n\n"
        f"[CÂU HỎI]\n{question}\n\n"
        f"[NGUYÊN VĂN TRANG TÀI LIỆU — bằng chứng gốc]\n{reference_text}\n\n"
        f"[CÂU TRẢ LỜI CỦA HỆ THỐNG CẦN CHẤM]\n{answer_text}\n\n"
        "Trả lời ĐÚNG theo định dạng sau, không thêm chữ nào khác:\n"
        "NHÃN: <một trong 3 nhãn ở trên, giữ nguyên dấu ngoặc kép>\n"
        "LÝ DO: <1 câu ngắn giải thích>"
    )


def parse_grading_response(text: str) -> tuple:
    label_match = re.search(r'NHÃN:\s*"?([^"\n]+)"?', text)
    reason_match = re.search(r"LÝ DO:\s*(.+)", text, re.DOTALL)
    label = label_match.group(1).strip() if label_match else ""
    reason = reason_match.group(1).strip() if reason_match else text.strip()

    # Chuẩn hóa nhãn về đúng 1 trong 3 giá trị chuẩn, tránh lệch chính tả nhỏ
    normalized = None
    for valid in VALID_LABELS:
        if valid in label.lower():
            normalized = valid
            break
    if normalized is None:
        normalized = f"[KHÔNG PHÂN LOẠI ĐƯỢC: {label}]"
    return normalized, reason


def main():
    parser = argparse.ArgumentParser(description="Chấm answer_correctness_manual bằng Gemini + nguyên văn PDF.")
    parser.add_argument("--results", required=True, help="CSV đầu vào (kết quả run_dev_qa_v2.py)")
    parser.add_argument("--dev-set", default="data/eval_sets/dev_questions_normalized.json")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--out", default=None, help="CSV đầu ra. Mặc định: <results>_graded.csv")
    parser.add_argument("--model", default=GRADING_MODEL)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--force", action="store_true",
                         help="Chấm lại cả các dòng đã có sẵn answer_correctness_manual (mặc định bỏ qua).")
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    out_path = Path(args.out).resolve() if args.out else results_path.with_name(
        results_path.stem + "_graded.csv"
    )

    with open(Path(args.dev_set).resolve(), "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    questions_by_id = {q["id"]: q for q in dev_data["questions"]}

    with open(results_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    if "answer_correctness_reason" not in fieldnames:
        fieldnames.append("answer_correctness_reason")
        for r in rows:
            r.setdefault("answer_correctness_reason", "")

    ensure_configured()
    model = genai.GenerativeModel(args.model, generation_config={"temperature": GRADING_TEMPERATURE})

    n_graded = 0
    for r in rows:
        if str(r.get("is_answerable", "")).strip().lower() != "true":
            continue
        if str(r.get("is_abstained", "")).strip().lower() == "true":
            continue
        if str(r.get("is_error", "")).strip().lower() == "true":
            continue
        if not r.get("answer_text", "").strip():
            continue
        if r.get("answer_correctness_manual", "").strip() and not args.force:
            continue

        q = questions_by_id.get(r["id"])
        if q is None:
            print(f"[WARN] Không tìm thấy {r['id']} trong dev-set, bỏ qua.")
            continue

        raw_pages = [int(p) for p in (q.get("expected_page") or [])]
        if raw_pages:
            lo = max(1, min(raw_pages) - 1)
            hi = max(raw_pages) + 1
            expected_pages = list(range(lo, hi + 1))
        else:
            expected_pages = []
        if not expected_pages:
            print(f"[WARN] {r['id']} không có expected_page, bỏ qua (không có bằng chứng gốc để chấm).")
            continue

        reference_text = extract_reference_pages(args.corpus_dir, q["source_file"], expected_pages)
        prompt = build_grading_prompt(q["question"], reference_text, r["answer_text"])

        try:
            response = model.generate_content(prompt)
            label, reason = parse_grading_response(response.text)
        except Exception as exc:
            print(f"[ERROR] {r['id']}: lỗi gọi Gemini chấm điểm: {exc}")
            continue

        r["answer_correctness_manual"] = label
        r["answer_correctness_reason"] = reason
        n_graded += 1
        print(f"  {r['id']:8s} -> {label}  ({reason[:80]})")
        time.sleep(args.sleep)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nĐã chấm {n_graded} câu, ghi ra: {out_path}")

    # Thống kê nhanh để đối chiếu với bảng Tuần 4
    labels_count: Dict[str, int] = {}
    for r in rows:
        label = r.get("answer_correctness_manual", "").strip()
        if label:
            labels_count[label] = labels_count.get(label, 0) + 1
    if labels_count:
        print("\n[TỔNG HỢP answer_correctness_manual]")
        for label, count in labels_count.items():
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
"""
oracle_context_test.py — Kiểm tra "Oracle Context" cho 8 câu nghi vấn (Trạm 2)

BỐI CẢNH
--------
8 câu: dev_05, dev_06, dev_10, dev_16, dev_18, dev_19, dev_23, dev_25 đã được xác
nhận là ANSWERABLE thật sự (đối chiếu text thô + ảnh chụp gốc + answer_reference
trong dev_questions_normalized.json), nhưng khi chạy qua pipeline RAG đầy đủ
(retrieval tự động), hệ thống trả lời sai/thiếu ("một phần") hoặc từ chối hẳn
tùy theo ngưỡng τ.

Câu hỏi cần trả lời cho từng câu: NẾU đưa thẳng đúng trang tài liệu (bỏ qua hoàn
toàn retrieval), model có trả lời đúng không?
    - Vẫn sai / vẫn từ chối dù có đúng context  -> lỗi GENERATION (lỗi ở model)
    - Trả lời đúng khi có đúng context          -> lỗi RETRIEVAL/CHUNKING (hệ thống
                                                    gốc không tìm/tách đúng đoạn,
                                                    không phải lỗi ở model)

BẢN CẬP NHẬT SO VỚI PHIÊN BẢN TRƯỚC
------------------------------------
Phiên bản trước tự đoán prompt và tự gọi Gemini API riêng — có nguy cơ so sánh
không công bằng nếu prompt/model config đoán sai. Bản này dùng THẲNG
source/qa/qa_generator.py thật (build_qa_prompt + generate_answer), nên prompt,
model, temperature, cơ chế retry đều là ĐÚNG HỆT hệ thống production. Cách duy
nhất "giả" ở đây là: thay vì để retrieval (FAISS/BM25) tự tìm chunk, ta tự tay
nhét đúng trang gold vào dưới dạng SearchHit với score=1.0 (luôn vượt ngưỡng τ),
để build_qa_prompt() không bao giờ abstain vì threshold — CHỈ CÒN LẠI khả năng
model tự abstain thật sự (model_refusal), đúng biến cần đo.

LƯU Ý QUAN TRỌNG VỀ 2 LOẠI "TỪ CHỐI" (đọc kỹ qa_generator.py để hiểu):
  - "Không tìm thấy thông tin LIÊN QUAN trong tài liệu." -> abstain_reason=
    "retrieval_threshold" (do lọc điểm số, model CHƯA từng được gọi). Trong bài
    test này, luồng abstain kiểu này bị vô hiệu hoá chủ đích (score=1.0, tau=0.0).
  - "Không tìm thấy thông tin trong tài liệu." (không có "liên quan") -> abstain_
    reason="model_refusal", do CHÍNH MODEL trả lời đúng câu này sau khi đã đọc
    chunk thật. Đây mới là tín hiệu ta cần: nếu xuất hiện trong oracle test dù
    đã đưa đúng trang chứa đáp án -> lỗi generation thật sự.

CÁCH LÀM
--------
1. Với mỗi câu, lấy answer_pages trong dev_questions_normalized.json, mở rộng
   thêm PAGE_BUFFER trang mỗi bên (mặc định 1) — phòng answer_pages ghi thiếu
   (nghi ngờ dev_19: answer_pages=[28] nhưng đáp án phân bổ có thể ở trang 29).
2. Dùng build_clean_pages() (ingest_glue.py thật) để lấy text đã chuẩn hoá.
3. Bọc mỗi trang vào 1 SearchHit giả (score=1.0, is_bridge=False).
4. Gọi generate_answer(question, hits, tau=0.0, ...) — HÀM PRODUCTION THẬT,
   không viết lại prompt hay gọi API riêng.
5. Ghi kết quả + chẩn đoán ra CSV.

VẪN CẦN BẠN TỰ LÀM
-------------------
1. Việc chấm "đúng/sai/một phần" cuối cùng vẫn cần đọc answer_text so với
   answer_reference bằng mắt — heuristic_keyword_overlap() chỉ cảnh báo nhanh
   bằng số liệu/năm trùng khớp, KHÔNG phải chấm điểm chính thức.
2. Nếu về sau qa_generator.py đổi cách abstain (câu chữ, ngưỡng...), cập nhật
   lại MODEL_ABSTAIN_TEXT / logic tương ứng trong chính qa_generator.py — script
   này gọi thẳng hàm nên sẽ tự động ăn theo, không cần sửa gì ở đây.

CÁCH CHẠY
---------
    export GEMINI_API_KEY="..."   (hoặc để trong file .env, load_dotenv() đã có
                                    sẵn trong qa_generator.py)
    python oracle_context_test.py \
        --dev-questions dev_questions_normalized.json \
        --corpus-dir data/corpus \
        --output results/oracle_context_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# Dùng đúng hàm ghép trang sạch của pipeline thật.
from source.retrieval.ingest_glue import build_clean_pages
from source.ingestion.pdf_loader import PDFLoadError
from source.retrieval.vectorstore import SearchHit

# Dùng THẲNG module QA thật — không viết lại prompt / lời gọi model riêng.
from source.qa.qa_generator import generate_answer, DEFAULT_MODEL, MODEL_ABSTAIN_TEXT

# ----- Cấu hình cho lần chạy oracle-context này -----
TARGET_QUESTION_IDS = [
    "dev_05", "dev_06", "dev_10", "dev_16",
    "dev_18", "dev_19", "dev_23", "dev_25",
]

# Mở rộng thêm N trang mỗi bên quanh answer_pages đã ghi trong dev set, để không
# lặp lại đúng lỗi "bỏ sót trang" mà chính hệ thống RAG đang mắc (vd nghi ngờ dev_19
# cần thêm trang 29 dù answer_pages chỉ ghi [28]).
PAGE_BUFFER = 1

# score giả gán cho mỗi chunk oracle — chỉ cần >= mọi giá trị tau hợp lý (0..1)
# để build_qa_prompt() không bao giờ abstain vì threshold trong bài test này.
ORACLE_SCORE = 1.0
ORACLE_TAU = 0.0


@dataclass
class OracleTarget:
    question_id: str
    source_file: str
    question: str
    answer_reference: str
    gold_pages: List[int]      # answer_pages gốc từ dev set
    oracle_pages: List[int]    # gold_pages đã mở rộng thêm PAGE_BUFFER


@dataclass
class OracleResult:
    question_id: str
    source_file: str
    oracle_pages_used: str
    answer_text: str
    is_abstained: bool
    abstain_reason: str
    is_error: bool
    error_message: str
    heuristic_match: Optional[bool]  # None nếu is_abstained hoặc is_error
    diagnosis: str                   # generation_error | retrieval_or_chunking_error_likely | needs_manual_review
    latency_seconds: Optional[float]
    model_used: Optional[str]


def load_targets(dev_questions_path: Path, question_ids: List[str]) -> List[OracleTarget]:
    data = json.loads(dev_questions_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in data["questions"]}

    targets: List[OracleTarget] = []
    for qid in question_ids:
        if qid not in by_id:
            print(f"[WARN] Không tìm thấy '{qid}' trong {dev_questions_path}, bỏ qua.")
            continue
        q = by_id[qid]
        gold_pages = q.get("answer_pages") or []
        if not gold_pages:
            print(f"[WARN] '{qid}' không có answer_pages, bỏ qua.")
            continue

        oracle_pages = _expand_page_range(gold_pages, PAGE_BUFFER)

        targets.append(
            OracleTarget(
                question_id=qid,
                source_file=q["source_file"],
                question=q["question"],
                answer_reference=q.get("answer_reference", ""),
                gold_pages=gold_pages,
                oracle_pages=oracle_pages,
            )
        )
    return targets


def _expand_page_range(pages: List[int], buffer: int) -> List[int]:
    """Lấy nguyên đoạn [min-buffer, max+buffer], không +-buffer từng trang lẻ,
    để chắc chắn không hở trang nào ở giữa nếu pages không liên tục."""
    lo = max(1, min(pages) - buffer)
    hi = max(pages) + buffer
    return list(range(lo, hi + 1))


def build_oracle_hits(target: OracleTarget, corpus_dir: Path) -> List[SearchHit]:
    """Lấy text đã chuẩn hoá của các trang oracle_pages, bọc thành SearchHit giả
    (score=1.0) để đưa thẳng vào build_qa_prompt()/generate_answer() thật —
    không qua FAISS/BM25."""
    pdf_path = corpus_dir / target.source_file
    try:
        clean_pages = build_clean_pages(str(pdf_path))
    except PDFLoadError as exc:
        raise RuntimeError(f"Không đọc được '{pdf_path}': {exc}") from exc

    pages_by_number = {p["page_number"]: p for p in clean_pages}

    hits: List[SearchHit] = []
    for page_num in target.oracle_pages:
        page = pages_by_number.get(page_num)
        if page is None:
            continue  # trang ngoài phạm vi file
        hits.append(
            SearchHit(
                chunk_id=f"oracle_{target.question_id}_p{page_num}",
                source_file=target.source_file,
                page_number=page_num,
                page_range=None,
                is_bridge=False,
                text=page["text"],
                score=ORACLE_SCORE,
            )
        )
    return hits


def heuristic_keyword_overlap(answer_text: str, answer_reference: str) -> bool:
    """CẢNH BÁO NHANH, KHÔNG PHẢI chấm điểm chính thức — xem docstring đầu file."""
    signal_tokens = re.findall(r"\d[\d.,%/-]*", answer_reference)
    if not signal_tokens:
        return True  # không có số liệu cụ thể để so khớp -> không cảnh báo, cần đọc tay
    lowered = answer_text.lower()
    hits = sum(1 for tok in signal_tokens if tok.lower() in lowered)
    return hits / len(signal_tokens) >= 0.6


def run_oracle_test(dev_questions_path: Path, corpus_dir: Path, output_path: Path) -> None:
    targets = load_targets(dev_questions_path, TARGET_QUESTION_IDS)
    results: List[OracleResult] = []

    for target in targets:
        print(f"[{target.question_id}] Đang xử lý (trang oracle: {target.oracle_pages})...")

        try:
            hits = build_oracle_hits(target, corpus_dir)
        except RuntimeError as exc:
            results.append(
                OracleResult(
                    question_id=target.question_id,
                    source_file=target.source_file,
                    oracle_pages_used=str(target.oracle_pages),
                    answer_text="",
                    is_abstained=False,
                    abstain_reason="",
                    is_error=True,
                    error_message=str(exc),
                    heuristic_match=None,
                    diagnosis="needs_manual_review",
                    latency_seconds=None,
                    model_used=None,
                )
            )
            continue

        if not hits:
            print(f"  [WARN] Không lấy được trang nào cho {target.question_id}, bỏ qua.")
            continue

        # Gọi THẲNG hàm production thật — prompt, model, retry đều y hệt hệ thống.
        answer = generate_answer(target.question, hits, tau=ORACLE_TAU, target_model=DEFAULT_MODEL)

        if answer.is_error:
            diagnosis = "needs_manual_review"
            heuristic = None
        elif answer.is_abstained:
            # Đã đưa đúng trang chứa đáp án vào tay, model VẪN trả lời đúng câu
            # MODEL_ABSTAIN_TEXT -> lỗi nằm ở generation, không phải retrieval.
            diagnosis = "generation_error"
            heuristic = None
        else:
            heuristic = heuristic_keyword_overlap(answer.answer_text, target.answer_reference)
            diagnosis = "retrieval_or_chunking_error_likely"

        results.append(
            OracleResult(
                question_id=target.question_id,
                source_file=target.source_file,
                oracle_pages_used=str(target.oracle_pages),
                answer_text=answer.answer_text,
                is_abstained=answer.is_abstained,
                abstain_reason=answer.abstain_reason or "",
                is_error=answer.is_error,
                error_message=answer.error_message or "",
                heuristic_match=heuristic,
                diagnosis=diagnosis,
                latency_seconds=answer.latency_seconds,
                model_used=answer.model_used,
            )
        )

    _write_csv(results, output_path)
    _print_summary(results)


def _write_csv(results: List[OracleResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nĐã ghi kết quả: {output_path}")


def _print_summary(results: List[OracleResult]) -> None:
    print("\n" + "=" * 60)
    print("TÓM TẮT CHẨN ĐOÁN ORACLE-CONTEXT")
    print("=" * 60)
    for r in results:
        flag = ""
        if r.diagnosis == "retrieval_or_chunking_error_likely" and r.heuristic_match is False:
            flag = "  <-- heuristic KHÔNG khớp, đọc kỹ bằng tay"
        print(f"  {r.question_id:10s} | {r.diagnosis:32s}{flag}")
    print("\nLưu ý: 'retrieval_or_chunking_error_likely' vẫn cần đối chiếu answer_text")
    print("với answer_reference bằng mắt trước khi kết luận chính thức trong báo cáo.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-questions", type=Path, default=Path("dev_questions_normalized.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--output", type=Path, default=Path("results/oracle_context_results.csv"))
    args = parser.parse_args()

    run_oracle_test(args.dev_questions, args.corpus_dir, args.output)


if __name__ == "__main__":
    main()
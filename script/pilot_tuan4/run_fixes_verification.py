"""
run_fixes_verification.py — GỘP TẤT CẢ VÀO 1 FILE DUY NHẤT.

CHỈ import từ các file ĐÃ CÓ SẴN trong source/ (pdf_loader, text_normalizer,
chunker, vectorstore, qa_generator) — KHÔNG cần thêm file nào khác vào dự án,
KHÔNG sửa bất kỳ file gốc nào. Toàn bộ phần "mới" (nhận diện trang biểu đồ,
render ảnh, gọi model kèm ảnh) được viết thẳng trong file này.

HAI CÁCH SỬA ĐƯỢC ÁP DỤNG:
  Phần 1 — Tăng kích thước chunk (256 -> 320 token) để giảm lỗi CẮT CỤT
           (dev_10, dev_23). Không sửa chunker.py — chỉ ghi đè hằng số
           MAX_TOKENS_PER_CHUNK của module đó lúc chạy (monkeypatch trong bộ
           nhớ, không đụng file trên đĩa).
  Phần 2 — Gửi kèm ẢNH trang cho các trang "nhiều biểu đồ" để giảm lỗi ĐỌC
           NHẦM SỐ LIỆU (dev_16). Tự viết hàm gọi model kèm ảnh ngay trong
           file này (không sửa qa_generator.py).

BẢN CẬP NHẬT (sau --calibrate-pdf): heuristic tự động nhận diện "trang nhiều
biểu đồ" dựa trên vector_object_count KHÔNG dùng được — đã thử trên
normal_vinamilkbaocao2014_53tr.pdf và phát hiện 44/53 trang đều vượt ngưỡng
40, vì cả tài liệu dùng phong cách infographic (viền/icon trang trí ở mọi
trang), không riêng gì trang có biểu đồ số liệu thật. Đáng chú ý hơn: trang 7
(biết chắc có lỗi thật, xem dev_16 Tuần 4) chỉ đạt vector_object_count=179 -
THẤP hơn nhiều trang chữ bình thường khác (vd trang 8 = 11.500) -> không có
ngưỡng số nào tách bạch được loại trang cần tìm. Đã chuyển sang danh sách thủ
công CHART_HEAVY_PAGES_MANUAL: phạm vi vấn đề đã xác nhận qua oracle-context
test Tuần 4 chỉ có DUY NHẤT trang 7 (file Vinamilk) thực sự cần gửi kèm ảnh,
ảnh hưởng cả dev_05 lẫn dev_16 (cùng nằm trang 7) - không cần bộ nhận diện tự
động phức tạp cho 1 trường hợp duy nhất.

Đây là test RETRIEVAL THẬT (không phải oracle-context): để retrieval tự đi
tìm chunk như bình thường, xem 2 cách sửa trên có giúp hệ thống TỰ TÌM RA câu
trả lời hay không.

CÁCH CHẠY (chạy bình thường, làm cả 2 cách sửa + test retrieval):
    python run_fixes_verification.py \
        --dev-questions data/eval_sets/dev_questions_normalized.json \
        --corpus-dir data/corpus \
        --vncorenlp-dir ./vncorenlp_models \
        --output results/fixes_verification_v2.csv

CÁCH CHẠY (chỉ in bảng vector_object_count để tham khảo, KHÔNG dùng để quyết
định ngưỡng tự động nữa - đã xác nhận cách đó không dùng được, giữ lại option
này chỉ để tiện tra cứu thủ công khi cần mở rộng CHART_HEAVY_PAGES_MANUAL cho
file khác sau này):
    python run_fixes_verification.py --calibrate-pdf data/corpus/ten_file.pdf

Lần chạy đầu có thể chậm (tải tokenizer + model embedding qua mạng lần đầu).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set

import pdfplumber
import google.generativeai as genai
from PIL import Image

from source.ingestion.pdf_loader import load_pdf_pages, PDFLoadError, PageData  # noqa: E402
from source.ingestion.text_normalizer import normalize_page_text
from source.retrieval import chunker as chunker_module
from source.retrieval.vectorstore import ChunkIndex, embed_chunks, search as vs_search, SearchHit
from source.qa.qa_generator import (
    DEFAULT_TAU,
    DEFAULT_MODEL,
    GENERATION_TEMPERATURE,
    MAX_RATE_LIMIT_RETRIES,
    RATE_LIMIT_RETRY_SECONDS,
    QAAnswer,
    build_qa_prompt,
    generate_answer as generate_answer_text_only,
    _ensure_configured,
    _is_rate_limit_error,
    _is_model_abstain_text,
    _build_citations,
)

TARGET_QUESTION_IDS = [
    "dev_05", "dev_06", "dev_10", "dev_16",
    "dev_18", "dev_19", "dev_23", "dev_25",
]

# ----- Phần 1: kích thước chunk mới -----
NEW_MAX_TOKENS_PER_CHUNK = 320  # gốc là 256

# ----- Phần 2: danh sách thủ công trang cần gửi kèm ảnh -----
# Xem giải thích đầy đủ ở docstring đầu file: heuristic tự động (vector_object_
# count) đã thử và KHÔNG dùng được cho tài liệu infographic này. Thêm
# ("tên_file.pdf", số_trang) vào set này nếu phát hiện thêm trường hợp khác
# cần gửi kèm ảnh (qua oracle-context test hoặc đọc lỗi thực tế), không đoán
# trước hàng loạt.
CHART_HEAVY_PAGES_MANUAL = {
    ("normal_vinamilkbaocao2014_53tr.pdf", 7),
}

RENDER_RESOLUTION = 220  # tăng từ 150 -> 220 để model đọc đúng nhãn/số liệu chart nhỏ hơn

RETRIEVAL_K = 5  # có thể ghi đè bằng --retrieval-k khi chạy

# In ra trang + preview text của TỪNG hit sau mỗi lần search, để mắt thường
# kiểm tra xem trang chứa biểu đồ/thông tin cần thiết có lọt vào top-k hay
# không. Bật/tắt bằng --debug-retrieval.
DEBUG_RETRIEVAL = False


def _hit_text_preview(hit, max_len: int = 90) -> str:
    """Lấy text preview của 1 SearchHit, thử vài tên field phổ biến vì
    không chắc field text thật tên gì trong SearchHit gốc."""
    for attr in ("text", "chunk_text", "content", "raw_text"):
        val = getattr(hit, attr, None)
        if val:
            s = str(val).replace("\n", " ").strip()
            return s[:max_len] + ("..." if len(s) > max_len else "")
    return "(không tìm thấy field text trên SearchHit — xem lại tên field trong vectorstore.py)"


def _print_debug_hits(question_id: str, hits: List[SearchHit]) -> None:
    print(f"    [DEBUG {question_id}] Top-{len(hits)} hits:")
    for rank, h in enumerate(hits, start=1):
        page = h.page_range if h.is_bridge else h.page_number
        print(f"      #{rank} chunk_id={h.chunk_id} page={page} | {_hit_text_preview(h)!r}")


# =====================================================================
# PHẦN 2 — nhận diện trang cần ảnh (danh sách thủ công) + render ảnh
# =====================================================================

def get_chart_heavy_page_numbers(pages: List[PageData]) -> Set[int]:
    """Trả về set số trang cần gửi kèm ảnh, tra theo (source_file, page_number)
    trong CHART_HEAVY_PAGES_MANUAL. Xem lý do bỏ heuristic tự động ở docstring
    đầu file."""
    if not pages:
        return set()
    source_file = pages[0].source_file
    return {
        p.page_number
        for p in pages
        if (source_file, p.page_number) in CHART_HEAVY_PAGES_MANUAL
    }


def render_page_images(file_path: str, page_numbers: Set[int]) -> Dict[int, bytes]:
    images: Dict[int, bytes] = {}
    if not page_numbers:
        return images
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num in page_numbers:
                idx = page_num - 1
                if idx < 0 or idx >= len(pdf.pages):
                    continue
                try:
                    page_image = pdf.pages[idx].to_image(resolution=RENDER_RESOLUTION)
                    buffer = io.BytesIO()
                    page_image.original.save(buffer, format="PNG")
                    images[page_num] = buffer.getvalue()
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Không render được ảnh trang {page_num}: {exc}")
    except Exception as exc:
        print(f"[WARN] Không mở được '{file_path}' để render ảnh: {exc}")
    return images


# =====================================================================
# PHẦN 2 (tiếp) — gọi model kèm ảnh, viết lại tối thiểu dựa trên
# generate_answer() gốc trong qa_generator.py (copy phần retry/parse kết quả,
# chỉ thêm nhánh gửi kèm ảnh)
# =====================================================================

def generate_answer_multimodal(
    question: str,
    hits: List[SearchHit],
    page_image_by_hit: Dict[str, bytes],
    tau: float = DEFAULT_TAU,
    target_model: str = DEFAULT_MODEL,
) -> QAAnswer:
    build_result = build_qa_prompt(question, hits, tau=tau)

    if build_result.should_abstain:
        return QAAnswer(
            answer_text="Không tìm thấy thông tin liên quan trong tài liệu.",
            is_abstained=True,
            abstain_reason="retrieval_threshold",
            citations=[],
        )

    # Lấy ảnh (không trùng lặp) của các chunk đang dùng.
    seen_ids = set()
    image_bytes_list: List[bytes] = []
    for c in build_result.used_chunks:
        png = page_image_by_hit.get(c.chunk_id)
        if png and id(png) not in seen_ids:
            seen_ids.add(id(png))
            image_bytes_list.append(png)

    if not image_bytes_list:
        # Không liên quan tới trang biểu đồ -> dùng nguyên hàm gốc, đảm bảo
        # hành vi giống hệt production cho các câu không bị ảnh hưởng.
        return generate_answer_text_only(question, hits, tau=tau, target_model=target_model)

    _ensure_configured()
    model_name = target_model.replace("models/", "")

    content: list = [
        build_result.prompt
        + "\n\nLƯU Ý: một số trang tài liệu có biểu đồ mà phần trích xuất văn "
        "bản có thể bị xáo trộn thứ tự số liệu. Ảnh chụp nguyên trang được đính "
        "kèm bên dưới — hãy ưu tiên đọc số liệu trực tiếp từ ảnh khi nó liên "
        "quan tới câu hỏi, thay vì chỉ dựa vào văn bản đã trích."
    ]
    for png_bytes in image_bytes_list:
        content.append(Image.open(io.BytesIO(png_bytes)))

    retry_count = 0
    while True:
        try:
            model = genai.GenerativeModel(model_name, generation_config={"temperature": GENERATION_TEMPERATURE})
            start = time.time()
            response = model.generate_content(content)
            elapsed = time.time() - start

            um = getattr(response, "usage_metadata", None)
            answer_text = response.text
            model_abstained = _is_model_abstain_text(answer_text)
            return QAAnswer(
                answer_text=answer_text,
                is_abstained=model_abstained,
                abstain_reason="model_refusal" if model_abstained else None,
                citations=[] if model_abstained else _build_citations(build_result.used_chunks),
                latency_seconds=round(elapsed, 3),
                prompt_tokens=getattr(um, "prompt_token_count", None),
                output_tokens=getattr(um, "candidates_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
                model_used=model_name,
            )
        except Exception as e:
            if _is_rate_limit_error(e) and retry_count < MAX_RATE_LIMIT_RETRIES:
                retry_count += 1
                print(f"\n[RATE LIMIT 429] Đợi {RATE_LIMIT_RETRY_SECONDS}s, thử lại {retry_count}/{MAX_RATE_LIMIT_RETRIES}...")
                time.sleep(RATE_LIMIT_RETRY_SECONDS)
                continue
            return QAAnswer(
                answer_text="", is_abstained=False,
                citations=_build_citations(build_result.used_chunks),
                is_error=True, error_message=str(e), model_used=model_name,
            )


# =====================================================================
# PHẦN 1 — ghép trang sạch + chia chunk kích thước mới
# =====================================================================

def build_clean_pages_with_images(pdf_path: Path) -> List[dict]:
    """Tương đương ingest_glue.build_clean_pages(), viết lại thẳng ở đây (chỉ
    gọi các hàm có sẵn từ pdf_loader/text_normalizer) + thêm ảnh biểu đồ."""
    raw_pages = load_pdf_pages(str(pdf_path))
    chart_page_numbers = get_chart_heavy_page_numbers(raw_pages)
    images = render_page_images(str(pdf_path), chart_page_numbers)

    clean_pages = []
    for page in raw_pages:
        result = normalize_page_text(page.raw_text)
        clean_pages.append(
            {
                "page_number": page.page_number,
                "source_file": page.source_file,
                "text": result.normalized_text,
                "chart_image_png": images.get(page.page_number),
            }
        )
    return clean_pages


def build_chunks_with_new_size(clean_pages: List[dict]) -> List[dict]:
    original_value = chunker_module.MAX_TOKENS_PER_CHUNK
    chunker_module.MAX_TOKENS_PER_CHUNK = NEW_MAX_TOKENS_PER_CHUNK
    try:
        pages_for_chunker = [
            {"page_number": p["page_number"], "source_file": p["source_file"], "text": p["text"]}
            for p in clean_pages
        ]
        chunks = chunker_module.chunk_by_page(pages_for_chunker)
    finally:
        chunker_module.MAX_TOKENS_PER_CHUNK = original_value

    image_by_page = {p["page_number"]: p["chart_image_png"] for p in clean_pages}
    for c in chunks:
        c["page_image_png"] = None
        if not c["is_bridge"] and c.get("page_number") in image_by_page:
            c["page_image_png"] = image_by_page[c["page_number"]]

    return chunks


# =====================================================================
# Chạy chính
# =====================================================================

@dataclass
class VerificationResult:
    question_id: str
    source_file: str
    question: str
    retrieved_pages: str
    used_chart_image: bool
    answer_text: str
    answer_reference: str
    is_abstained: bool
    abstain_reason: str
    is_error: bool
    error_message: str


def run_verification(dev_questions_path: Path, corpus_dir: Path, vncorenlp_dir: str, output_path: Path) -> None:
    data = json.loads(dev_questions_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in data["questions"]}
    targets = [by_id[qid] for qid in TARGET_QUESTION_IDS if qid in by_id]

    by_source_file: Dict[str, List[dict]] = {}
    for q in targets:
        by_source_file.setdefault(q["source_file"], []).append(q)

    results: List[VerificationResult] = []

    for source_file, questions in by_source_file.items():
        pdf_path = corpus_dir / source_file
        print(f"\n=== Xử lý file: {source_file} ===")

        try:
            clean_pages = build_clean_pages_with_images(pdf_path)
        except PDFLoadError as exc:
            print(f"[LỖI] Không đọc được '{pdf_path}': {exc}")
            continue

        chart_count = sum(1 for p in clean_pages if p["chart_image_png"])
        print(f"  Số trang được gửi kèm ảnh (danh sách thủ công): {chart_count}")

        print("  Đang chia chunk (kích thước mới)...")
        chunks = build_chunks_with_new_size(clean_pages)
        print(f"  Tổng số chunk: {len(chunks)}")

        print("  Đang tính embedding + dựng index (có thể chậm lần đầu)...")
        vectors = embed_chunks(chunks, vncorenlp_dir)
        index = ChunkIndex(dim=vectors.shape[1] if vectors.shape[0] else 768)
        index.add(vectors, chunks, vncorenlp_dir=vncorenlp_dir)

        # Map chunk_id -> ảnh (nếu có) để tra lại sau khi search trả về SearchHit
        image_by_chunk_id = {c["chunk_id"]: c["page_image_png"] for c in chunks if c["page_image_png"]}

        for q in questions:
            print(f"  [{q['id']}] Đang tìm kiếm + sinh câu trả lời...")
            hits = vs_search(index, q["question"], vncorenlp_dir, k=RETRIEVAL_K)

            if DEBUG_RETRIEVAL:
                _print_debug_hits(q["id"], hits)

            used_chart_image = any(h.chunk_id in image_by_chunk_id for h in hits)
            retrieved_pages = ", ".join(
                str(h.page_range) if h.is_bridge else str(h.page_number) for h in hits
            )

            answer = generate_answer_multimodal(q["question"], hits, image_by_chunk_id, tau=DEFAULT_TAU)

            results.append(
                VerificationResult(
                    question_id=q["id"],
                    source_file=source_file,
                    question=q["question"],
                    retrieved_pages=retrieved_pages,
                    used_chart_image=used_chart_image,
                    answer_text=answer.answer_text,
                    answer_reference=q.get("answer_reference", ""),
                    is_abstained=answer.is_abstained,
                    abstain_reason=answer.abstain_reason or "",
                    is_error=answer.is_error,
                    error_message=answer.error_message or "",
                )
            )

    _write_csv(results, output_path)
    _print_summary(results)


def _write_csv(results: List[VerificationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nĐã ghi kết quả: {output_path}")


def _print_summary(results: List[VerificationResult]) -> None:
    print("\n" + "=" * 70)
    print("KẾT QUẢ SAU KHI ÁP DỤNG 2 CÁCH SỬA (đọc bằng mắt, so với answer_reference)")
    print("=" * 70)
    for r in results:
        img_flag = " [dùng ảnh trang]" if r.used_chart_image else ""
        print(f"\n--- {r.question_id}{img_flag} — trang tìm được: {r.retrieved_pages} ---")
        print(f"Model trả lời : {r.answer_text}")
        print(f"Đáp án chuẩn  : {r.answer_reference}")
        if r.is_abstained:
            print(f"[TỪ CHỐI — abstain_reason={r.abstain_reason}]")
        if r.is_error:
            print(f"[LỖI GỌI MODEL: {r.error_message}]")


def run_calibration(pdf_path: Path) -> None:
    """CHỈ in bảng vector_object_count/char_count từng trang rồi dừng — tham
    khảo thủ công khi cần thêm trang vào CHART_HEAVY_PAGES_MANUAL cho file
    khác. KHÔNG dùng số liệu này để tự động chọn ngưỡng - đã xác nhận cách đó
    không tách bạch được (xem docstring đầu file)."""
    pages = load_pdf_pages(str(pdf_path))
    print(f"\n{'Trang':>6} | {'char_count':>10} | {'image_count':>11} | {'vector_object_count':>19}")
    print("-" * 56)
    for p in pages:
        print(f"{p.page_number:>6} | {p.char_count:>10} | {p.image_count:>11} | {p.vector_object_count:>19}")
    print(
        "\nBảng này chỉ để THAM KHẢO khi bạn tự xem xét thêm trang nào vào "
        "CHART_HEAVY_PAGES_MANUAL - KHÔNG dùng để tự động chọn ngưỡng số "
        "(đã xác nhận không tách bạch được trang biểu đồ thật khỏi trang chữ "
        "thường trong tài liệu infographic dạng này)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-questions", type=Path, default=Path("data/eval_sets/dev_questions_normalized.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--vncorenlp-dir", type=str, default="./vncorenlp_models")
    parser.add_argument("--output", type=Path, default=Path("results/fixes_verification_v2.csv"))
    parser.add_argument(
        "--calibrate-pdf", type=Path, default=None,
        help="Nếu truyền, CHỈ in bảng vector_object_count để tham khảo thủ công rồi dừng, không chạy gì khác.",
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=None,
        help="Ghi đè RETRIEVAL_K (mặc định 5). Dùng để thử k lớn hơn xem trang đúng có lọt vào top-k hay không.",
    )
    parser.add_argument(
        "--debug-retrieval", action="store_true",
        help="In ra trang + preview text của từng hit sau mỗi lần search.",
    )
    args = parser.parse_args()

    global RETRIEVAL_K, DEBUG_RETRIEVAL
    if args.retrieval_k is not None:
        RETRIEVAL_K = args.retrieval_k
    if args.debug_retrieval:
        DEBUG_RETRIEVAL = True

    # QUAN TRỌNG: resolve thành đường dẫn TUYỆT ĐỐI ngay từ đầu, trước khi JVM
    # (khởi động bên trong embed_chunks -> py_vncorenlp) có cơ hội đổi CWD của
    # tiến trình. Nếu không, path tương đối của corpus_dir sẽ bị sai đối với
    # các file PDF xử lý SAU file đầu tiên trong danh sách.
    args.dev_questions = args.dev_questions.resolve()
    args.corpus_dir = args.corpus_dir.resolve()
    args.vncorenlp_dir = str(Path(args.vncorenlp_dir).resolve())
    args.output = args.output.resolve()

    if args.calibrate_pdf is not None:
        run_calibration(args.calibrate_pdf.resolve())
        return

    run_verification(args.dev_questions, args.corpus_dir, args.vncorenlp_dir, args.output)


if __name__ == "__main__":
    main()
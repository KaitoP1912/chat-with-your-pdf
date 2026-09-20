"""Debug retrieval: chạy đúng pipeline search() đang dùng trong dự án.

Mục tiêu:
- Nhận vào 1 file PDF và 1 câu hỏi.
- Dùng cùng pipeline hiện tại: build_clean_pages -> chunk_by_page -> build_index -> search(..., k=15)
- In ra từng chunk kèm metadata + nội dung đầy đủ để đánh giá bằng mắt.
- Lưu log sang file markdown/text để xem sau.

Không sửa logic retrieval hoặc app_ui.py/qa_generator.py.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from source.retrieval.ingest_glue import build_clean_pages  # noqa: E402
from source.retrieval.chunker import chunk_by_page  # noqa: E402
from source.retrieval.vectorstore import build_index, search  # noqa: E402

VNCORENLP_DIR = str(PROJECT_ROOT / "vncorenlp_models")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "debug_retrieval"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug retrieval: chạy đúng pipeline search() hiện tại cho 1 PDF và 1 câu hỏi.\n"
            "Mặc định dùng tau=0.38 và k=15 như config.py đang khóa."
        )
    )
    parser.add_argument("pdf_path", help="Đường dẫn tới file PDF cần index")
    parser.add_argument("question", help="Câu hỏi cần debug retrieval")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="(Tuỳ chọn) đường dẫn file markdown/text để lưu kết quả; nếu để trống thì tự tạo trong results/debug_retrieval/",
    )
    return parser.parse_args()


def _build_hits(pdf_path: str, question: str, k: int) -> list:
    pages = build_clean_pages(pdf_path)
    chunks = chunk_by_page(pages)
    index = build_index(chunks, VNCORENLP_DIR)
    hits = search(index, question, VNCORENLP_DIR, k=k)
    return hits


def _render_markdown(question: str, pdf_path: str, hits: list) -> str:
    lines: list[str] = []
    lines.append("# Retrieval Debug Report")
    lines.append("")
    lines.append(f"- File PDF: `{pdf_path}`")
    lines.append(f"- Câu hỏi: `{question}`")
    lines.append(f"- tau (config): `{config.TAU}`")
    lines.append(f"- k (search): `{config.TOP_K_GENERATION}`")
    lines.append(f"- search() gọi: `search(index, question, VNCORENLP_DIR, k=config.TOP_K_GENERATION)`")
    lines.append("")
    lines.append("> Lưu ý: `tau` là ngưỡng abstention/guardrail ở tầng generation; `search()` hiện tại chỉ nhận `k` và không chứa logic tau.\n")

    if not hits:
        lines.append("## Kết quả: không có chunk nào trả về.")
        return "\n".join(lines) + "\n"

    for idx, hit in enumerate(hits, start=1):
        page_label = "(không xác định)"
        if hit.page_number is not None:
            page_label = f"trang {hit.page_number}"
        elif hit.page_range:
            page_label = f"trang {hit.page_range}"

        lines.append(f"## #{idx} | chunk_id: `{hit.chunk_id}`")
        lines.append(f"- page_number: `{hit.page_number}`")
        lines.append(f"- page_range: `{hit.page_range}`")
        lines.append(f"- is_bridge: `{hit.is_bridge}`")
        lines.append(f"- page_label: `{page_label}`")
        lines.append(f"- cosine_similarity (score): `{hit.score:.6f}`")
        if hit.bm25_score is not None:
            lines.append(f"- bm25_score: `{hit.bm25_score:.6f}`")
        if hit.rrf_score is not None:
            lines.append(f"- rrf_score: `{hit.rrf_score:.6f}`")
        lines.append("")
        lines.append("### Text chunk")
        lines.append("```text")
        lines.append(hit.text.strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    pdf_path = str(Path(args.pdf_path).resolve())
    question = args.question.strip()

    if not question:
        print("Câu hỏi rỗng. Hãy truyền 1 câu hỏi hợp lệ.")
        sys.exit(2)

    print(f"[1/3] Build index cho PDF: {pdf_path}")
    print(f"[2/3] Search cho câu hỏi: {question!r}")
    print(f"[3/3] k={config.TOP_K_GENERATION}, tau={config.TAU}")

    hits = _build_hits(pdf_path, question, k=config.TOP_K_GENERATION)

    print("\n=== KẾT QUẢ SEARCH() ===")
    if not hits:
        print("Không có chunk nào được trả về.")
    else:
        for idx, hit in enumerate(hits, start=1):
            page_label = f"trang {hit.page_number}" if hit.page_number is not None else (f"trang {hit.page_range}" if hit.page_range else "(không xác định)")
            print(f"\n#{idx} | chunk_id={hit.chunk_id} | {page_label} | bridge={hit.is_bridge} | cosine={hit.score:.6f}")
            print("--- TEXT START ---")
            print(hit.text.strip())
            print("--- TEXT END ---")

    # lưu markdown
    markdown_text = _render_markdown(question, pdf_path, hits)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"retrieval_debug_{timestamp}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")
    print(f"\n[Đã lưu kết quả debug tại: {output_path}]")


if __name__ == "__main__":
    main()

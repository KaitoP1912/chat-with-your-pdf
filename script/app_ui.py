"""
script/app_ui.py — Giao diện MVP "Chat with Your PDF" (Việc A, Tuần 6).

Refactor từ bản gốc `app_streamlit.py` (đặt sai vị trí ở gốc repo) — CHỈ
tổ chức lại code cho gọn và sửa 1 lỗi runtime (xem CHANGELOG cuối file),
KHÔNG đổi bất kỳ hành vi RAG nào. File này CHỈ đóng vai trò lớp giao diện:
mọi logic xử lý (đọc PDF, chuẩn hóa, chia đoạn, embedding, tìm kiếm, sinh
câu trả lời) dùng lại NGUYÊN VẸN các hàm đã có trong source/ — giống hệt
cách app_cli.py đã làm, không viết lại logic bên trong.

Phần CSS/markup thuần trình bày đã được tách sang script/app_ui_style.py
để file này chỉ còn logic pipeline + luồng UI (dễ đọc/review hơn).

Tuân thủ đúng yêu cầu đề cương "ứng dụng tải lên MỘT tài liệu mỗi phiên":
mỗi phiên Streamlit chỉ giữ 1 file PDF đang làm việc trong session_state,
muốn đổi file phải bấm "Tải file khác".

Tham số ĐÃ KHÓA đọc từ config.py (KHÔNG cho người dùng chỉnh trên giao diện):
  - tau = config.TAU              (0.38)
  - k   = config.TOP_K_GENERATION (15)
  - model = config.MODEL_NAME     (gemini-3.5-flash-lite)
  - chunking = chunk_by_page (kiến trúc mặc định, KHÔNG dùng chunk_fixed_size)

Cách chạy (từ thư mục gốc project, nơi có source/, config.py, .env):
    python -m streamlit run script/app_ui.py --server.fileWatcherType none

Yêu cầu: file .env ở thư mục gốc project có GEMINI_API_KEY hợp lệ (đúng
biến mà qa_generator.py đang đọc qua os.getenv("GEMINI_API_KEY")).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  (đọc tham số đã khóa: TAU, TOP_K_GENERATION, MODEL_NAME)

from source.ingestion.pdf_loader import load_pdf_pages, PDFLoadError  # noqa: E402
from source.ingestion.scan_detector import detect_scan, should_reject, DocScanStatus  # noqa: E402
from source.retrieval.ingest_glue import build_clean_pages  # noqa: E402
from source.retrieval.chunker import chunk_by_page  # noqa: E402
from source.retrieval.vectorstore import build_index, search  # noqa: E402
from source.qa.qa_generator import generate_answer  # noqa: E402

from script.app_ui_style import inject_css  # noqa: E402

VNCORENLP_DIR = os.path.abspath("./vncorenlp_models")


# =====================================================================
# HELPERS THUẦN (không phụ thuộc st.session_state)
# =====================================================================

def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return "(không có)"
    labels = []
    for c in citations:
        if c.get("page_number") is not None:
            labels.append(f"trang {c['page_number']}" + (" [bridge]" if c.get("is_bridge") else ""))
        elif c.get("page_range"):
            labels.append(f"trang {c['page_range']} [bridge liên trang]")
    seen = []
    for lb in labels:
        if lb not in seen:
            seen.append(lb)
    return ", ".join(seen) if seen else "(không xác định được trang)"


# =====================================================================
# LOGIC PIPELINE (session_state) — dùng nguyên hàm từ source/
# =====================================================================

def _reset_session() -> None:
    for key in ("doc_hash", "doc_name", "pdf_path", "index", "chunks", "scan_summary",
                "corpus_dir", "chat_history"):
        st.session_state.pop(key, None)


def _build_index_for_upload(uploaded_file) -> None:
    """Đọc PDF upload -> lưu tạm -> build_clean_pages -> chunk_by_page -> build_index.

    Ghi thẳng vào st.session_state, không return, để giữ đúng 1 tài liệu/phiên.
    Nếu file (theo hash nội dung) đã build trong phiên này rồi thì bỏ qua,
    tái sử dụng index cũ — tránh dựng lại index tốn thời gian khi người dùng
    hỏi tiếp câu thứ 2 trên cùng 1 file.
    """
    file_bytes = uploaded_file.getvalue()
    doc_hash = _file_hash(file_bytes)

    if st.session_state.get("doc_hash") == doc_hash:
        return  # đã build rồi (VD do Streamlit rerun), khỏi build lại

    _reset_session()

    tmp_dir = tempfile.mkdtemp(prefix="cwyp_")
    pdf_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    with st.spinner("Đang đọc PDF..."):
        try:
            raw_pages = load_pdf_pages(pdf_path)
        except PDFLoadError as e:
            st.error(f"Không xử lý được file: {e}")
            return

        scan_result = detect_scan(raw_pages)
        if should_reject(scan_result):
            st.error(
                f"File '{uploaded_file.name}' là PDF scan hoàn toàn (không có lớp text) — "
                "ứng dụng hiện chỉ hỗ trợ PDF có text layer. Vui lòng dùng OCR trước khi tải lên."
            )
            return

        pages = build_clean_pages(pdf_path)

    with st.spinner("Đang chia đoạn văn bản..."):
        chunks = chunk_by_page(pages)

    with st.spinner("Đang tạo embedding và dựng chỉ mục tìm kiếm (lần đầu có thể mất 30-60s)..."):
        index = build_index(chunks, VNCORENLP_DIR)

    st.session_state["doc_hash"] = doc_hash
    st.session_state["doc_name"] = uploaded_file.name
    st.session_state["pdf_path"] = pdf_path
    st.session_state["corpus_dir"] = tmp_dir
    st.session_state["index"] = index
    st.session_state["chunks"] = chunks
    st.session_state["scan_summary"] = scan_result
    st.session_state["chat_history"] = []


# =====================================================================
# CÁC MẢNH GIAO DIỆN (chạm st.session_state, nên giữ ở đây thay vì
# app_ui_style.py — style module không phụ thuộc trạng thái ứng dụng)
# =====================================================================

def _render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">📄</div>
                <div>
                    <div class="brand-name">Chat with Your PDF</div>
                    <div class="brand-desc">
                        Hỏi đáp thông minh với tài liệu PDF tiếng Việt
                    </div>
                </div>
            </div>
            <div class="brand-pill">RAG · Gemini</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_document_card() -> None:
    """Hiện thẻ tài liệu đang active + cảnh báo trang scan (nếu có)."""
    doc_name = st.session_state.get("doc_name")
    if not doc_name:
        return

    st.markdown(
        f"""
        <div class="active-doc">
            <div class="active-doc-left">
                <div class="active-doc-icon">📄</div>
                <div>
                    <div class="active-doc-name">{doc_name}</div>
                    <div class="active-doc-ready">● Sẵn sàng để hỏi</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scan_result = st.session_state.get("scan_summary")
    if scan_result and scan_result.doc_status != DocScanStatus.TEXT:
        st.warning(
            f"Phát hiện {len(scan_result.scan_pages)} trang scan/không đọc được text "
            f"(bỏ qua khi tìm kiếm): {scan_result.scan_pages or '(không có)'}"
        )


def _render_suggestions() -> None:
    st.markdown('<div class="starter-label">Gợi ý để bắt đầu</div>', unsafe_allow_html=True)

    suggestions = [
        ("📌", "Tóm tắt những nội dung chính của tài liệu"),
        ("🔎", "Tìm những thông tin quan trọng trong tài liệu"),
        ("💡", "Cho tôi biết các điểm đáng chú ý nhất"),
    ]

    cols = st.columns(3)
    for col, (icon, label) in zip(cols, suggestions):
        with col:
            st.markdown(
                f"""
                <div class="starter">
                    <div class="starter-icon">{icon}</div>
                    <div class="starter-text">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_sources(citations_label: str) -> None:
    """Render nhãn trích dẫn thành các "chip" nguồn gọn."""
    if not citations_label or citations_label == "(không có)":
        return

    parts = [part.strip() for part in citations_label.split(",") if part.strip()]
    if not parts:
        return

    chips = "".join(f'<span class="source-chip">📄 {part}</span>' for part in parts)
    st.markdown(f'<div class="source-label">Nguồn tham khảo</div>{chips}', unsafe_allow_html=True)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-title">📚 Tài liệu</div>
                <div class="sidebar-subtitle">
                    Một tài liệu PDF cho mỗi phiên làm việc.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        doc_name = st.session_state.get("doc_name")

        if doc_name:
            st.markdown(
                f"""
                <div class="sidebar-file">
                    <div class="sidebar-file-name">📄 {doc_name}</div>
                    <div class="sidebar-ready">✓ Đã xử lý và sẵn sàng</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            if st.button("↻  Tải file khác", use_container_width=True):
                _reset_session()
                st.rerun()
        else:
            st.info("Chưa có tài liệu nào.")

        st.divider()

        with st.expander("⚙️ Thông tin hệ thống"):
            st.caption(f"Model: `{config.MODEL_NAME}`")
            st.caption(f"Abstention τ: `{config.TAU}`")
            st.caption(f"Retrieval k: `{config.TOP_K_GENERATION}`")
            st.caption("Chunking: `page_aware`")

        st.caption("Lịch sử chat chỉ được giữ trong phiên hiện tại.")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    # CHỈ gọi set_page_config MỘT LẦN trong toàn bộ script (yêu cầu bắt buộc
    # của Streamlit — gọi 2 lần sẽ ném StreamlitAPIException). Bản gốc
    # app_streamlit.py gọi hàm này 2 lần (1 lần ở top-level module, 1 lần ở
    # đây) — đã gộp về đúng 1 lần khi refactor, xem CHANGELOG cuối file.
    st.set_page_config(
        page_title="Chat with Your PDF",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    _render_sidebar()
    _render_header()

    # ------------------------------------------------------------
    # CHƯA CÓ TÀI LIỆU: màn hình upload
    # ------------------------------------------------------------
    if not st.session_state.get("doc_name"):
        st.markdown(
            """
            <div class="upload-shell">
                <div class="upload-eyebrow">AI DOCUMENT ASSISTANT</div>
                <h1 class="upload-title">Trò chuyện với tài liệu<br>của bạn.</h1>
                <div class="upload-subtitle">
                    Tải lên một PDF tiếng Việt, sau đó đặt câu hỏi bằng ngôn ngữ tự nhiên.
                    Hệ thống sẽ tìm thông tin liên quan và trả lời kèm nguồn tham khảo.
                </div>
                <div class="upload-panel">
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Chọn file PDF",
            type=["pdf"],
            label_visibility="collapsed",
            help="PDF có text layer. Ứng dụng hiện hỗ trợ tối đa 60 trang.",
        )

        st.markdown(
            """
                </div>
                <div class="upload-hint">
                    PDF có text layer&nbsp; • &nbsp;tối đa 60 trang&nbsp; • &nbsp;1 tài liệu / phiên
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > 20 * 1024 * 1024:
                st.error("File vượt quá giới hạn 20MB của ứng dụng. Vui lòng chọn file nhỏ hơn.")
                st.stop()

            _build_index_for_upload(uploaded_file)

            if st.session_state.get("doc_name"):
                st.rerun()

        st.stop()

    # ------------------------------------------------------------
    # ĐÃ CÓ TÀI LIỆU: giao diện chat
    # ------------------------------------------------------------
    _render_document_card()

    if not st.session_state["chat_history"]:
        st.markdown(
            """
            <div class="welcome">
                <div class="welcome-icon">✨</div>
                <div class="welcome-title">Tài liệu đã sẵn sàng</div>
                <div class="welcome-text">
                    Hãy hỏi bất cứ điều gì về tài liệu. Câu trả lời sẽ được
                    tìm kiếm từ nội dung PDF và đi kèm nguồn tham khảo.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_suggestions()

    # Lịch sử chat
    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            if turn.get("is_error"):
                st.error(turn["answer_text"])
            else:
                st.write(turn["answer_text"])

            if not turn.get("is_abstained") and not turn.get("is_error"):
                _render_sources(turn.get("citations_label", ""))

            if turn.get("chart_pages_sent"):
                st.caption(f"🖼️ Đã gửi kèm ảnh trang: {', '.join(turn['chart_pages_sent'])}")

            meta_bits = []
            if turn.get("latency_seconds") is not None:
                meta_bits.append(f"{turn['latency_seconds']}s")
            if turn.get("total_tokens") is not None:
                meta_bits.append(f"{turn['total_tokens']} token")
            if meta_bits:
                st.caption(" · ".join(meta_bits))

    # ------------------------------------------------------------
    # LUỒNG HỎI-ĐÁP RAG — gọi nguyên hàm từ source/, tham số đã khóa
    # ------------------------------------------------------------
    question = st.chat_input("Hỏi bất cứ điều gì về tài liệu…")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và sinh câu trả lời…"):
                index = st.session_state["index"]

                hits = search(index, question, VNCORENLP_DIR, k=config.TOP_K_GENERATION)

                answer = generate_answer(
                    question,
                    hits,
                    tau=config.TAU,
                    target_model=config.MODEL_NAME,
                    corpus_dir=st.session_state["corpus_dir"],
                )

            if answer.is_error:
                st.error(f"Lỗi khi gọi Gemini: {answer.error_message}")
                citations_label = "(không có)"
            else:
                st.write(answer.answer_text)
                citations_label = _format_citations(answer.citations)

                if not answer.is_abstained:
                    _render_sources(citations_label)
                else:
                    st.caption("ℹ️ Không có nguồn đủ tin cậy để xác nhận câu trả lời.")

                if answer.chart_pages_sent:
                    st.caption(f"🖼️ Đã gửi kèm ảnh trang: {', '.join(answer.chart_pages_sent)}")

                meta_bits = []
                if answer.latency_seconds is not None:
                    meta_bits.append(f"{answer.latency_seconds}s")
                if answer.total_tokens is not None:
                    meta_bits.append(f"{answer.total_tokens} token")
                if meta_bits:
                    st.caption(" · ".join(meta_bits))

            st.session_state["chat_history"].append(
                {
                    "question": question,
                    "answer_text": answer.answer_text if not answer.is_error else f"[Lỗi] {answer.error_message}",
                    "is_abstained": answer.is_abstained,
                    "is_error": answer.is_error,
                    "citations_label": citations_label,
                    "chart_pages_sent": answer.chart_pages_sent,
                    "latency_seconds": answer.latency_seconds,
                    "total_tokens": answer.total_tokens,
                }
            )


if __name__ == "__main__":
    main()


# =====================================================================
# CHANGELOG (refactor từ app_streamlit.py gốc -> script/app_ui.py)
# =====================================================================
# 1. Vị trí: chuyển từ gốc repo (app_streamlit.py) -> script/app_ui.py,
#    đúng vị trí đã thống nhất cho Việc A, Tuần 6.
# 2. Sửa lỗi: xóa bỏ lệnh st.set_page_config() bị gọi 2 lần (1 lần ở
#    top-level module, 1 lần trong main()) — Streamlit chỉ cho phép gọi
#    đúng 1 lần/phiên, gọi 2 lần ném StreamlitAPIException. Giữ lại bản
#    trong main() (đầy đủ tham số hơn: layout="wide",
#    initial_sidebar_state="expanded").
# 3. Tách CSS/markup thuần (~350 dòng) sang script/app_ui_style.py — file
#    này chỉ còn logic pipeline + luồng UI, không đổi bất kỳ hành vi RAG
#    nào (build_clean_pages, chunk_by_page, build_index, search,
#    generate_answer vẫn được gọi y hệt bản gốc, cùng tham số đã khóa).
# 4. Xóa import `time` không dùng tới (dead code trong bản gốc).
# 5. KHÔNG đổi: mọi logic session_state, luồng cache theo doc_hash, luồng
#    xử lý scan-detection, luồng chat_history — giữ nguyên 100% so với
#    bản gốc app_streamlit.py.

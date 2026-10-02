"""Offline unit tests for the core document pipeline decisions."""
import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from source.ingestion.pdf_loader import PageData
from source.ingestion.scan_detector import (
    DocScanStatus,
    PageScanStatus,
    classify_page,
    detect_scan,
    should_reject,
)
from source.retrieval import vectorstore
from source.retrieval.vectorstore import ChunkIndex


@dataclass
class FakeSearchHit:
    chunk_id: str
    source_file: str
    page_number: int
    page_range: str | None
    is_bridge: bool
    text: str
    score: float


def load_filter_hits_by_threshold():
    """Load the pure filter while stubbing QA's optional Gemini import."""
    if "source.retrieval.vectorstore" not in sys.modules:
        vectorstore_module = ModuleType("source.retrieval.vectorstore")
        vectorstore_module.SearchHit = FakeSearchHit
        sys.modules["source.retrieval.vectorstore"] = vectorstore_module

    if "google.genai" not in sys.modules:
        genai_module = ModuleType("google.genai")
        types_module = ModuleType("google.genai.types")
        genai_module.types = types_module
        sys.modules["google.genai"] = genai_module
        sys.modules["google.genai.types"] = types_module

    from source.qa.qa_generator import filter_hits_by_threshold

    return filter_hits_by_threshold


def make_page(page_number, text="", image_count=0, vector_object_count=0):
    return PageData(
        page_number=page_number,
        source_file="fake.pdf",
        total_pages=4,
        raw_text=text,
        image_count=image_count,
        vector_object_count=vector_object_count,
    )


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (make_page(1, text="a" * 100), PageScanStatus.TEXT),
        (make_page(2, text="a" * 20), PageScanStatus.LOW_TEXT),
        (make_page(3, image_count=1), PageScanStatus.SCAN),
        (make_page(4), PageScanStatus.EMPTY),
    ],
)
def test_classify_page_returns_expected_status(page, expected):
    assert classify_page(page).status == expected


@pytest.mark.parametrize(
    ("pages", "expected_status", "expected_reject"),
    [
        ([make_page(1, text="a" * 100), make_page(2, text="a" * 100)], DocScanStatus.TEXT, False),
        ([make_page(1, text="a" * 100), make_page(2, image_count=1)], DocScanStatus.MIXED_SCAN, False),
        ([make_page(1, image_count=1), make_page(2, image_count=1)], DocScanStatus.FULL_SCAN, True),
        ([make_page(1, text="a" * 100), make_page(2)], DocScanStatus.TEXT, False),
    ],
)
def test_document_scan_status_controls_rejection(pages, expected_status, expected_reject):
    result = detect_scan(pages)
    assert result.doc_status == expected_status
    assert should_reject(result) is expected_reject


def test_build_clean_pages_omits_detected_scan_text_and_preserves_page_positions(
    monkeypatch,
):
    from source.retrieval import ingest_glue
    from source.retrieval.chunker import chunk_by_page

    raw_pages = [
        make_page(1, text="Nội dung hợp lệ trên trang một. " * 8),
        make_page(2, text="scan noise", image_count=1),
        make_page(3, text="Nội dung hợp lệ trên trang ba. " * 8),
    ]
    scan_result = detect_scan(raw_pages)
    monkeypatch.setattr(ingest_glue, "load_pdf_pages", lambda _: raw_pages)

    clean_pages = ingest_glue.build_clean_pages("fake.pdf", scan_result=scan_result)
    chunks = chunk_by_page(clean_pages, token_counter=lambda text: len(text.split()))

    assert [page["page_number"] for page in clean_pages] == [1, 2, 3]
    assert clean_pages[1]["text"] == ""
    assert all(chunk["page_number"] != 2 for chunk in chunks)
    assert not any(chunk["page_range"] in ("1-2", "2-3") for chunk in chunks)


def test_page_aware_chunker_only_bridges_two_nonempty_pages():
    from source.retrieval.chunker import chunk_by_page

    def page(number, text):
        return {"page_number": number, "source_file": "fake.pdf", "text": text}

    page_one = " ".join(f"trangmot{i}" for i in range(200))
    page_two = " ".join(f"tranghai{i}" for i in range(200))
    counter = lambda text: len(text.split())

    with_content = chunk_by_page([page(1, page_one), page(2, page_two)], token_counter=counter)
    assert any(chunk["is_bridge"] and chunk["page_range"] == "1-2" for chunk in with_content)

    with_empty_page = chunk_by_page([page(1, page_one), page(2, "")], token_counter=counter)
    assert not any(chunk["is_bridge"] and chunk["page_range"] == "1-2" for chunk in with_empty_page)


@pytest.mark.parametrize(
    ("scores", "expected_scores"),
    [
        ([0.20, 0.38, 0.50], [0.38, 0.50]),
        ([], []),
    ],
)
def test_filter_hits_by_threshold_uses_inclusive_tau(scores, expected_scores):
    filter_hits_by_threshold = load_filter_hits_by_threshold()

    def hit(score):
        return FakeSearchHit(
            chunk_id=f"chunk-{score}",
            source_file="fake.pdf",
            page_number=1,
            page_range=None,
            is_bridge=False,
            text="fake text",
            score=score,
        )

    hits = [hit(score) for score in scores]
    kept = filter_hits_by_threshold(hits, tau=0.38)
    assert [item.score for item in kept] == expected_scores


def test_filter_hits_preserves_input_order_without_mutating_hits():
    filter_hits_by_threshold = load_filter_hits_by_threshold()

    hits = [
        FakeSearchHit("a", "fake.pdf", 1, None, False, "a", 0.50),
        FakeSearchHit("b", "fake.pdf", 1, None, False, "b", 0.40),
        FakeSearchHit("c", "fake.pdf", 1, None, False, "c", 0.60),
    ]
    original_ids = [hit.chunk_id for hit in hits]

    kept = filter_hits_by_threshold(hits, tau=0.40)

    assert [hit.chunk_id for hit in kept] == ["a", "b", "c"]
    assert [hit.chunk_id for hit in hits] == original_ids


@pytest.mark.parametrize(
    ("bm25_scores", "expected_chunk_ids"),
    [
        ([0.0, 2.0, 0.0], ["chunk-1"]),
        ([0.0, 0.0, 0.0], []),
    ],
)
def test_search_hybrid_excludes_zero_score_bm25_chunks(
    monkeypatch, bm25_scores, expected_chunk_ids
):
    index = ChunkIndex.__new__(ChunkIndex)
    index._index = SimpleNamespace(ntotal=3)
    index._metadatas = [
        {
            "chunk_id": f"chunk-{i}",
            "source_file": "fake.pdf",
            "page_number": i + 1,
            "page_range": None,
            "is_bridge": False,
            "text": f"text {i}",
        }
        for i in range(3)
    ]
    index._bm25 = SimpleNamespace(get_scores=lambda _: bm25_scores)
    index._vncorenlp_dir = "unused-by-mock"
    monkeypatch.setattr(vectorstore, "segment_text", lambda text, _: text)

    hits = index.search_hybrid("query", k=3, use_dense=False)

    assert [hit.chunk_id for hit in hits] == expected_chunk_ids
    assert all(hit.bm25_score > 0 for hit in hits)


def test_resolve_used_sources_maps_deduplicates_and_ignores_invalid_indexes():
    from source.qa.qa_generator import _resolve_used_sources

    hits = [
        FakeSearchHit("chunk-a", "fake.pdf", 1, None, False, "a", 0.9),
        FakeSearchHit("chunk-b", "fake.pdf", 2, None, False, "b", 0.8),
    ]

    used_hits, chunk_ids = _resolve_used_sources(
        hits, [2, "2", 99, "not-an-index", 1]
    )

    assert used_hits == [hits[1], hits[0]]
    assert chunk_ids == ["chunk-b", "chunk-a"]


def test_generate_answer_abstains_before_gemini_when_no_hit_meets_tau(monkeypatch):
    from source.qa import qa_generator

    monkeypatch.setattr(
        qa_generator,
        "_get_client",
        lambda: pytest.fail("Gemini must not be called when tau removes every hit"),
    )
    hit = FakeSearchHit("weak", "fake.pdf", 17, None, False, "weak evidence", 0.37)

    answer = qa_generator.generate_answer("question without evidence", [hit], tau=0.38)

    assert answer.is_abstained
    assert answer.abstain_reason == "retrieval_threshold"
    assert answer.citations == []


def test_citations_use_page_metadata_and_only_display_used_chunks():
    from source.qa.qa_generator import _build_citations
    from script.app_ui import _format_citations

    hits = [
        FakeSearchHit("used", "fake.pdf", 17, None, False, "used text", 0.8),
        FakeSearchHit("unused", "fake.pdf", 21, None, False, "unused text", 0.7),
    ]

    citations = _build_citations([hits[0]])

    assert citations[0]["page_number"] == 17
    assert _format_citations(citations, ["used"]) == "trang 17"
    assert _format_citations(citations, ["not-used"]) == "(không có)"


def test_model_abstention_recognizes_test_32_refusal_with_explanation():
    from source.qa.qa_generator import _is_model_abstain_text

    answer = (
        "Không tìm thấy thông tin trong tài liệu về đơn vị phối hợp triển khai "
        "và phương thức thực hiện của khảo sát."
    )

    assert _is_model_abstain_text(answer)


def test_model_abstention_does_not_flag_test_25_honest_partial_answer():
    from source.qa.qa_generator import _is_model_abstain_text

    answer = (
        "Dựa vào tài liệu, trong 12 ngày đêm cuối năm 1972 quân dân miền Bắc "
        "đã bắt sống 43 giặc lái. Tài liệu không cung cấp con số cụ thể về "
        "tổng số trong toàn bộ các lần chống chiến tranh phá hoại."
    )

    assert not _is_model_abstain_text(answer)
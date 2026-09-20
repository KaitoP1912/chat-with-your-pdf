"""Offline unit tests for the core document pipeline decisions."""
import sys
from dataclasses import dataclass
from types import ModuleType
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
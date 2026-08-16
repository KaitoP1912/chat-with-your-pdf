"""
vectorstore.py — Bước 5, 6 của Trạm 2

Embedding (vietnamese-bi-encoder) + L2-normalize + FAISS IndexFlatIP.

Model bkai-foundation-models/vietnamese-bi-encoder:
  - 768 chiều, max_seq_length=256, backbone PhoBERT-base-v2.
  - BẮT BUỘC input đã tách từ trước (dùng word_segmenter.segment_text()).
  - Lần chạy đầu tiên cần mạng để tự tải model (~500MB tùy phiên bản), sau đó cache lại.

FAISS IndexFlatIP: inner product trên vector đã L2-normalize = cosine similarity.
Index tạo mới mỗi phiên làm việc, không persist ra đĩa (đúng phạm vi đề cương).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import faiss

from source.retrieval.word_segmenter import segment_text

EMBED_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
EMBED_DIM = 768

_model = None  # singleton, tránh load lại model nhiều lần


def _get_model():
    """Load SentenceTransformer 1 lần duy nhất (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def embed_chunks(chunks: List[dict], vncorenlp_dir: str) -> np.ndarray:
    """Tách từ từng chunk -> encode bằng vietnamese-bi-encoder -> L2-normalize.

    chunks: list dict có key "text" (đúng output của chunker.chunk_by_page /
            chunk_fixed_size).
    Trả về ma trận (N, 768) đã L2-normalize, sẵn sàng add vào FAISS.
    """
    if not chunks:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    segmented_texts = [segment_text(c["text"], vncorenlp_dir) for c in chunks]

    model = _get_model()
    vectors = model.encode(segmented_texts, convert_to_numpy=True)
    vectors = vectors.astype(np.float32)
    return _l2_normalize(vectors)


def embed_query(query_text: str, vncorenlp_dir: str) -> np.ndarray:
    """Tách từ + encode 1 câu hỏi. Trả về vector (768,) đã L2-normalize."""
    segmented = segment_text(query_text, vncorenlp_dir)
    model = _get_model()
    vector = model.encode([segmented], convert_to_numpy=True).astype(np.float32)
    return _l2_normalize(vector)[0]


@dataclass
class SearchHit:
    chunk_id: str
    source_file: str
    page_number: Optional[int]
    page_range: Optional[str]
    is_bridge: bool
    text: str
    score: float


class ChunkIndex:
    """Gói FAISS IndexFlatIP + metadata mapping (FAISS chỉ trả về số nguyên,
    không tự nhớ chunk nào ứng với vị trí nào -> phải tự lưu list song song).
    """

    def __init__(self, dim: int = EMBED_DIM):
        self._index = faiss.IndexFlatIP(dim)
        self._metadatas: List[dict] = []

    def add(self, vectors: np.ndarray, chunks: List[dict]) -> None:
        assert vectors.shape[0] == len(chunks), "Số vector và số chunk phải khớp nhau"
        self._index.add(vectors)
        self._metadatas.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[SearchHit]:
        if self._index.ntotal == 0:
            return []
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, min(k, self._index.ntotal))

        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadatas[int(idx)]
            hits.append(
                SearchHit(
                    chunk_id=meta["chunk_id"],
                    source_file=meta["source_file"],
                    page_number=meta.get("page_number"),
                    page_range=meta.get("page_range"),
                    is_bridge=meta.get("is_bridge", False),
                    text=meta["text"],
                    score=float(score),
                )
            )
        return hits


def build_index(chunks: List[dict], vncorenlp_dir: str) -> ChunkIndex:
    """Tiện ích gộp: embed toàn bộ chunk rồi dựng index luôn."""
    vectors = embed_chunks(chunks, vncorenlp_dir)
    index = ChunkIndex(dim=vectors.shape[1] if vectors.shape[0] else EMBED_DIM)
    index.add(vectors, chunks)
    return index


def search(index: ChunkIndex, query_text: str, vncorenlp_dir: str, k: int = 3) -> List[SearchHit]:
    query_vector = embed_query(query_text, vncorenlp_dir)
    return index.search(query_vector, k=k)

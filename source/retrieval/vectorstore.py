"""
vectorstore.py — Bước 5, 6 của Trạm 2 (Nâng cấp Tuần 4: Hybrid BM25 + Dense RRF)

Embedding (vietnamese-bi-encoder) + FAISS IndexFlatIP kết hợp BM25 (rank-bm25).
Áp dụng Reciprocal Rank Fusion (RRF) để tối ưu xếp hạng cho các câu hỏi nhiều ý/bridge chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

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
    """Tách từ từng chunk -> encode bằng vietnamese-bi-encoder -> L2-normalize."""
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
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None


class ChunkIndex:
    """Gói FAISS IndexFlatIP + metadata mapping + BM25 Sparse Index."""

    def __init__(self, dim: int = EMBED_DIM):
        self._index = faiss.IndexFlatIP(dim)
        self._metadatas: List[dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._vncorenlp_dir: Optional[str] = None

    def add(self, vectors: np.ndarray, chunks: List[dict], vncorenlp_dir: Optional[str] = None) -> None:
        assert vectors.shape[0] == len(chunks), "Số vector và số chunk phải khớp nhau"
        self._index.add(vectors)
        self._metadatas.extend(chunks)
        self._vncorenlp_dir = vncorenlp_dir

        if vncorenlp_dir and chunks:
            corpus_tokens = [
                segment_text(c["text"], vncorenlp_dir).lower().split()
                for c in chunks
            ]
            self._bm25 = BM25Okapi(corpus_tokens)

    def search_dense(self, query_vector: np.ndarray, k: int = 3) -> List[SearchHit]:
        """Dense search thuần qua FAISS."""
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

    def search_hybrid(self, query_text: str, k: int = 3, rrf_k: int = 60) -> List[SearchHit]:
        """Hybrid search kết hợp Dense Vector và BM25 qua RRF."""
        if self._index.ntotal == 0:
            return []

        # 1. Dense search lấy thứ hạng
        q_vec = embed_query(query_text, self._vncorenlp_dir)
        dense_hits = self.search_dense(q_vec, k=len(self._metadatas))

        # 2. BM25 search
        q_tokens = segment_text(query_text, self._vncorenlp_dir).lower().split()
        bm25_scores = self._bm25.get_scores(q_tokens) if self._bm25 else [0.0] * len(self._metadatas)
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # 3. Hợp nhất RRF
        rrf_scores: Dict[str, float] = {}
        dense_map: Dict[str, SearchHit] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            dense_map[hit.chunk_id] = hit

        for rank, idx in enumerate(bm25_ranked_indices, start=1):
            c_id = self._metadatas[idx]["chunk_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + rank))

        # Sắp xếp theo RRF rank giảm dần
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        hits: List[SearchHit] = []
        for cid in sorted_chunk_ids[:k]:
            orig_hit = dense_map[cid]
            idx = next(i for i, m in enumerate(self._metadatas) if m["chunk_id"] == cid)
            hits.append(
                SearchHit(
                    chunk_id=orig_hit.chunk_id,
                    source_file=orig_hit.source_file,
                    page_number=orig_hit.page_number,
                    page_range=orig_hit.page_range,
                    is_bridge=orig_hit.is_bridge,
                    text=orig_hit.text,
                    score=orig_hit.score,  # Giữ score cosine gốc phục vụ Dense Guardrail
                    bm25_score=float(bm25_scores[idx]),
                    rrf_score=float(rrf_scores[cid]),
                )
            )
        return hits


def build_index(chunks: List[dict], vncorenlp_dir: str) -> ChunkIndex:
    """Embed toàn bộ chunk và dựng index Hybrid (FAISS + BM25)."""
    vectors = embed_chunks(chunks, vncorenlp_dir)
    index = ChunkIndex(dim=vectors.shape[1] if vectors.shape[0] else EMBED_DIM)
    index.add(vectors, chunks, vncorenlp_dir=vncorenlp_dir)
    return index


def search(index: ChunkIndex, query_text: str, vncorenlp_dir: str, k: int = 3) -> List[SearchHit]:
    """Tìm kiếm mặc định qua Hybrid Search."""
    return index.search_hybrid(query_text, k=k)
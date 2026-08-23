"""
test_query_decomposition_retrieval.py

Đánh giá Hybrid Search (BM25 + Dense) kết hợp Dense Guardrail Threshold (tau = 0.38):
- BM25 kéo chunk đúng của dev_13, dev_14 lên Top 1.
- Dense Guardrail chặn câu hỏi out_of_scope.
- TUYỆT ĐỐI KHÔNG GỌI GEMINI API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent

_vncorenlp_candidates = [
    PROJECT_ROOT / "models" / "wordsegmenter",
    PROJECT_ROOT / "vncorenlp_models",
]
VNCORENLP_DIR = None
for p in _vncorenlp_candidates:
    if p.exists():
        VNCORENLP_DIR = str(p.resolve())
        break

if VNCORENLP_DIR is None:
    VNCORENLP_DIR = str((PROJECT_ROOT / "models" / "wordsegmenter").resolve())

CORPUS_DIR = (PROJECT_ROOT / "data" / "corpus").resolve()
DEV_SET_PATH = (PROJECT_ROOT / "data" / "eval_sets" / "dev_questions_normalized.json").resolve()
TAU_THRESHOLD = 0.38  # Ngưỡng calibrated tối ưu cho bi-encoder

from source.retrieval.ingest_glue import build_clean_pages
from source.retrieval.chunker import chunk_by_page
from source.retrieval.vectorstore import build_index, embed_query, SearchHit, ChunkIndex
from source.retrieval.word_segmenter import segment_text


class HybridRetriever:
    def __init__(self, chunks: List[dict], dense_index: ChunkIndex, vncorenlp_dir: str):
        self.chunks = chunks
        self.dense_index = dense_index
        self.vncorenlp_dir = vncorenlp_dir

        self.corpus_tokens = [
            segment_text(c["text"], vncorenlp_dir).lower().split()
            for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search_hybrid(self, query: str, k: int = 3, rrf_k: int = 60) -> List[dict]:
        q_vec = embed_query(query, self.vncorenlp_dir)
        dense_hits = self.dense_index.search(q_vec, k=len(self.chunks))

        q_tokens = segment_text(query, self.vncorenlp_dir).lower().split()
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            chunk_map[hit.chunk_id] = {
                "chunk_id": hit.chunk_id,
                "page_number": hit.page_number,
                "page_range": hit.page_range,
                "is_bridge": hit.is_bridge,
                "dense_score": hit.score,
                "text": hit.text,
            }

        for rank, idx in enumerate(bm25_ranked_indices, start=1):
            c_id = self.chunks[idx]["chunk_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + rank))
            if c_id not in chunk_map:
                chunk_map[c_id] = {
                    "chunk_id": c_id,
                    "page_number": self.chunks[idx].get("page_number"),
                    "page_range": self.chunks[idx].get("page_range"),
                    "is_bridge": self.chunks[idx].get("is_bridge", False),
                    "dense_score": 0.0,
                    "text": self.chunks[idx]["text"],
                }
            chunk_map[c_id]["bm25_score"] = float(bm25_scores[idx])

        ranked_chunks = sorted(
            [{**chunk_map[cid], "rrf_score": rrf_scores[cid]} for cid in rrf_scores],
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return ranked_chunks[:k]


def run_experiment():
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"VnCoreNLP dir : {VNCORENLP_DIR}")
    print(f"Ngưỡng tau    : {TAU_THRESHOLD}\n")

    with open(DEV_SET_PATH, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    questions = dev_data["questions"]
    results = []

    file_groups: Dict[str, List[dict]] = {}
    for q in questions:
        file_groups.setdefault(q["source_file"], []).append(q)

    for source_file, q_list in file_groups.items():
        pdf_path = CORPUS_DIR / source_file
        print(f"--- Đang xử lý: {source_file} ({len(q_list)} câu) ---")
        pages = build_clean_pages(str(pdf_path))
        chunks = chunk_by_page(pages)
        dense_index = build_index(chunks, VNCORENLP_DIR)
        hybrid_retriever = HybridRetriever(chunks, dense_index, VNCORENLP_DIR)

        for q in q_list:
            expected = [int(p) for p in (q.get("expected_page") or [])] if q.get("expected_page") else []
            hits = hybrid_retriever.search_hybrid(q["question"], k=3)
            top1 = hits[0]

            def check_page_hit(hit_item):
                if not expected:
                    return False
                if hit_item.get("is_bridge") and hit_item.get("page_range"):
                    pages_in_range = [int(x) for x in hit_item["page_range"].split("-")]
                    return any(p in expected for p in pages_in_range)
                elif hit_item.get("page_number") is not None:
                    return hit_item["page_number"] in expected
                return False

            hit_at_3 = any(check_page_hit(h) for h in hits)
            passed_guardrail = top1.get("dense_score", 0.0) >= TAU_THRESHOLD

            results.append({
                "id": q["id"],
                "type": q.get("type", "answerable"),
                "is_answerable": q["is_answerable"],
                "dense_top1": round(top1.get("dense_score", 0.0), 4),
                "bm25_top1": round(top1.get("bm25_score", 0.0), 4),
                "hit_at_3": hit_at_3 if q["is_answerable"] else "",
                "passed_tau_0.38": passed_guardrail,
            })

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ VỚI NGƯỠNG CALIBRATED TAU = 0.38")
    print("=" * 80)
    print(df.to_string(index=False))

    ans = df[df["is_answerable"] == True]
    unans = df[df["is_answerable"] == False]

    n_ans_pass = sum(ans["passed_tau_0.38"])
    n_unans_blocked = sum(~unans["passed_tau_0.38"])

    print("\n" + "=" * 50)
    print(f"Answerable giữ lại được: {n_ans_pass}/{len(ans)} ({n_ans_pass/len(ans)*100:.1f}%)")
    print(f"Unanswerable bị chặn tại Retrieval: {n_unans_blocked}/{len(unans)} ({n_unans_blocked/len(unans)*100:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    run_experiment()
"""
word_segmenter.py — Bước 1 của Trạm 2

Tách từ tiếng Việt bằng py_vncorenlp, BẮT BUỘC trước khi đưa văn bản vào
model embedding vietnamese-bi-encoder (PhoBERT-based chỉ hiểu đúng khi input
đã tách từ, ví dụ "vui tính" -> "vui_tính").

Lưu ý quan trọng đã đo thực nghiệm:
  - Chỉ cần annotator 'wseg' (word segmentation), KHÔNG load thêm pos/ner/parse
    -> giảm thời gian khởi tạo model từ ~30s xuống ~0.7s.
  - py_vncorenlp dùng JVM (qua jnius) bên trong -> chỉ nên khởi tạo VnCoreNLP
    MỘT LẦN trong toàn bộ chương trình (singleton), không tạo lại nhiều lần.
"""

from __future__ import annotations

from typing import List, Optional

import py_vncorenlp

_segmenter: Optional[py_vncorenlp.VnCoreNLP] = None


def get_segmenter(vncorenlp_dir: str) -> py_vncorenlp.VnCoreNLP:
    """Trả về instance VnCoreNLP dùng chung (singleton) — chỉ load model 1 lần."""
    global _segmenter
    if _segmenter is None:
        _segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
    return _segmenter


def segment_text(text: str, vncorenlp_dir: str) -> str:
    """Tách từ 1 đoạn text, trả về 1 chuỗi duy nhất (nối các câu bằng khoảng trắng).

    py_vncorenlp.word_segment() trả về list[str] (mỗi câu 1 phần tử) -> nối lại
    thành 1 chuỗi vì chunk của mình không cần giữ ranh giới câu riêng.
    """
    if not text or not text.strip():
        return ""
    segmenter = get_segmenter(vncorenlp_dir)
    sentences: List[str] = segmenter.word_segment(text)
    return " ".join(sentences)


if __name__ == "__main__":
    import sys

    vncorenlp_dir = sys.argv[1] if len(sys.argv) > 1 else "./vncorenlp_models"
    sample = "Cô ấy là một người vui tính. Đại học Bách khoa Hà Nội đã phát triển các chương trình đào tạo."
    print("Input :", sample)
    print("Output:", segment_text(sample, vncorenlp_dir))

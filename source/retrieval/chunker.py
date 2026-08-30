"""
chunker.py — Bước 2, 3, 4 của Trạm 2

Hai chiến lược chia đoạn, dùng chung 1 cấu trúc metadata để so sánh công bằng
(theo đúng yêu cầu đề cương: "chênh lệch kết quả chỉ phản ánh đúng biến ranh giới chunk").

  - chunk_by_page(): chia theo trang, <=MAX_TOKENS_PER_CHUNK token/chunk, overlap ~30 từ,
    có thêm bridge chunk 128 từ mỗi bên tại MỖI ranh giới trang-trang VÀ
    tại mỗi ranh giới mảnh-mảng nội bộ trong cùng 1 trang (sửa Tuần 4,
    xem lý do chi tiết ở comment trong chunk_by_page()).
  - chunk_fixed_size(): baseline, chia cố định 170 từ/chunk toàn văn bản,
    không tôn trọng ranh giới trang, overlap 30 từ.

Đếm token: dùng tokenizer thật của model bkai-foundation-models/vietnamese-bi-encoder
qua hàm `default_token_counter()`. Hàm này CẦN MẠNG để tải tokenizer lần đầu
(không tải được trong môi trường sandbox hiện tại demo code này — sẽ tự chạy
được trên máy có internet). Có thể truyền `token_counter` khác vào để test
logic chia đoạn độc lập với việc đếm token (xem ví dụ ở cuối file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, TypedDict

# ----- Cấu hình đã chốt, không tự đổi -----
# MAX_TOKENS_PER_CHUNK tăng 256 -> 320 (Tuần 5, sau oracle-context test Tuần 4):
# dev_10, dev_23 bị trả lời thiếu dù đúng trang nằm trong top-k, do 1 trang dài
# bị cắt thành nhiều mảnh nhỏ, nội dung cần thiết (VD "chính trị" ở dev_10, điểm
# 4-5 ở dev_23) rơi vào mảnh khác không được xếp hạng đủ cao. Tăng kích thước
# mỗi mảnh giúp gộp nhiều nội dung liền mạch hơn vào 1 đoạn, giảm số lần phải
# cắt trang. Đã thử hướng khác (tăng k từ 3->5 giữ nguyên 256) trước đó nhưng
# KHÔNG hiệu quả (không cải thiện dev_10/dev_23, còn gây tác dụng phụ ở dev_25)
# -> đã loại bỏ hướng đó, xem lại kèm log so sánh trong báo cáo Tuần 5.
# Lưu ý: model embedding vietnamese-bi-encoder chỉ tính điểm dựa trên tối đa
# 256 token đầu (max_seq_length=256) dù đoạn dài hơn - phần vượt 256 token vẫn
# được lưu đầy đủ và gửi cho Gemini khi đoạn đó được chọn, nhưng KHÔNG ảnh
# hưởng tới việc xếp hạng tìm kiếm. Chọn 320 (không tăng quá xa 256) để hạn
# chế rủi ro giảm độ chính xác tìm kiếm.
MAX_TOKENS_PER_CHUNK = 320
OVERLAP_WORDS = 30
BRIDGE_WORDS_EACH_SIDE = 128
FIXED_CHUNK_WORDS = 170
EMBED_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"


class ChunkDict(TypedDict):
    chunk_id: str
    source_file: str
    page_number: Optional[int]     # None nếu là bridge chunk (dùng page_range thay)
    page_range: Optional[str]      # "N-N+1", chỉ có ở bridge chunk
    text: str
    token_count: int
    is_bridge: bool


TokenCounter = Callable[[str], int]


def default_token_counter() -> TokenCounter:
    """Trả về hàm đếm token bằng tokenizer thật của vietnamese-bi-encoder.

    CẦN MẠNG lần đầu gọi (tự tải tokenizer về cache). Nếu đang test offline,
    truyền token_counter khác (ví dụ đếm theo số từ) vào các hàm chunk_* bên dưới.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)

    def _count(text: str) -> int:
        return len(tokenizer.encode(text))

    return _count


def _split_words(text: str) -> List[str]:
    return text.split()


def _chunk_page_text(
    words: List[str],
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_words: int,
) -> List[str]:
    """Chia list từ của 1 trang thành các đoạn con <= max_tokens, overlap overlap_words.

    Chiến lược: tăng dần cửa sổ theo từ cho tới khi vượt max_tokens thì chốt đoạn,
    lùi lại overlap_words để bắt đầu đoạn tiếp theo.
    """
    if not words:
        return []

    pieces: List[str] = []
    start = 0
    n = len(words)

    while start < n:
        end = start
        # Mở rộng dần tới khi chạm giới hạn token (kiểm tra từng bước để không vượt quá)
        while end < n:
            candidate = " ".join(words[start:end + 1])
            if token_counter(candidate) > max_tokens:
                break
            end += 1
        if end == start:  # 1 từ đã vượt giới hạn (hiếm) -> vẫn phải lấy để tránh vòng lặp vô hạn
            end = start + 1

        pieces.append(" ".join(words[start:end]))

        if end >= n:
            break
        start = max(end - overlap_words, start + 1)  # lùi lại overlap, luôn tiến ít nhất 1 từ

    return pieces


def chunk_by_page(
    pages: List[dict],
    token_counter: Optional[TokenCounter] = None,
) -> List[ChunkDict]:
    """Chia theo trang (chiến lược đề xuất) + sinh thêm bridge chunk ở mỗi ranh giới trang.

    pages: list các dict {"page_number": int, "source_file": str, "text": str}
           (đúng output của source.retrieval.ingest_glue.build_clean_pages)
    """
    token_counter = token_counter or default_token_counter()
    chunks: List[ChunkDict] = []

    if not pages:
        return chunks

    source_file = pages[0]["source_file"]

    # --- Chunk thường theo từng trang + bridge chunk NỘI BỘ TRANG ---
    #
    # Sửa lỗi phát hiện ở Tuần 4 (dev_13, dev_14): khi 1 trang dài bị cắt
    # thành >=2 mảnh bởi _chunk_page_text() (vì vượt MAX_TOKENS_PER_CHUNK),
    # overlap OVERLAP_WORDS=30 từ giữa 2 mảnh liền kề là quá ít để giữ đủ tín
    # hiệu cho câu hỏi "bridge" hỏi gộp 2 ý nằm ở 2 mảnh khác nhau (vd Điều 94
    # + Điều 95 trên cùng 1 trang) -> cosine similarity bị pha loãng, rớt
    # ngưỡng tau dù thông tin có thật trong tài liệu. Trước đây bridge chunk
    # chỉ được sinh ở ranh giới TRANG-TRANG (vòng lặp bên dưới), không có ở
    # ranh giới MẢNH-MẢNH trong cùng 1 trang. Nay thêm bridge chunk tương tự
    # (BRIDGE_WORDS_EACH_SIDE từ mỗi bên) tại mỗi ranh giới mảnh nội bộ trang,
    # dùng chung logic "bỏ qua nếu 1 phía rỗng" như bridge trang-trang.
    for page in pages:
        words = _split_words(page["text"])
        pieces = _chunk_page_text(words, token_counter, MAX_TOKENS_PER_CHUNK, OVERLAP_WORDS)
        piece_word_lists = [_split_words(p) for p in pieces]

        for i, piece_text in enumerate(pieces):
            chunks.append(
                ChunkDict(
                    chunk_id=f"{source_file}_p{page['page_number']}_c{i}",
                    source_file=source_file,
                    page_number=page["page_number"],
                    page_range=None,
                    text=piece_text,
                    token_count=token_counter(piece_text),
                    is_bridge=False,
                )
            )

        # Bridge chunk nội bộ trang: tại mỗi ranh giới mảnh i / i+1 trong CÙNG 1 trang.
        for i in range(len(piece_word_lists) - 1):
            tail = piece_word_lists[i][-BRIDGE_WORDS_EACH_SIDE:]
            head = piece_word_lists[i + 1][:BRIDGE_WORDS_EACH_SIDE]

            if not tail or not head:
                continue

            bridge_text = " ".join(tail + head)
            chunks.append(
                ChunkDict(
                    chunk_id=f"{source_file}_p{page['page_number']}_intrabridge{i}",
                    source_file=source_file,
                    # Cùng 1 trang thật -> page_number rõ ràng, KHÔNG dùng
                    # page_range (page_range chỉ dành cho bridge trang-trang,
                    # nơi 2 trang khác nhau thật sự).
                    page_number=page["page_number"],
                    page_range=None,
                    text=bridge_text,
                    token_count=token_counter(bridge_text),
                    is_bridge=True,
                )
            )

    # --- Bridge chunk tại mỗi ranh giới trang N / N+1 ---
    for i in range(len(pages) - 1):
        page_n = pages[i]
        page_n1 = pages[i + 1]
        words_n = _split_words(page_n["text"])
        words_n1 = _split_words(page_n1["text"])

        tail = words_n[-BRIDGE_WORDS_EACH_SIDE:]
        head = words_n1[:BRIDGE_WORDS_EACH_SIDE]

        # Nếu 1 trong 2 phía rỗng (trang scan/trống, thường gặp ở file mixed-scan),
        # bridge chunk sẽ chỉ chứa nội dung 1 phía -> trùng lặp y hệt chunk thường
        # của trang đó, không có giá trị "nối 2 trang" -> bỏ qua, không sinh bridge rác.
        if not tail or not head:
            continue

        bridge_text = " ".join(tail + head)

        page_range = f"{page_n['page_number']}-{page_n1['page_number']}"
        chunks.append(
            ChunkDict(
                chunk_id=f"{source_file}_bridge_{page_range}",
                source_file=source_file,
                page_number=None,
                page_range=page_range,
                text=bridge_text,
                token_count=token_counter(bridge_text),
                is_bridge=True,
            )
        )

    return chunks


def chunk_fixed_size(
    pages: List[dict],
    token_counter: Optional[TokenCounter] = None,
) -> List[ChunkDict]:
    """Baseline: nối toàn bộ văn bản, chia cố định FIXED_CHUNK_WORDS từ/chunk,
    KHÔNG tôn trọng ranh giới trang. Gán page_number theo trang chiếm đa số ký tự.
    """
    token_counter = token_counter or default_token_counter()
    chunks: List[ChunkDict] = []

    if not pages:
        return chunks

    source_file = pages[0]["source_file"]

    # Ghép toàn văn bản, nhưng nhớ vị trí ký tự bắt đầu/kết thúc của mỗi trang
    # để sau này biết 1 đoạn chunk rơi vào (các) trang nào.
    full_text_parts: List[str] = []
    page_char_ranges: List[tuple] = []  # (page_number, start_char, end_char) trên full_text
    cursor = 0
    for page in pages:
        text = page["text"]
        start = cursor
        full_text_parts.append(text)
        cursor += len(text) + 1  # +1 cho khoảng trắng nối
        end = cursor - 1
        page_char_ranges.append((page["page_number"], start, end))
    full_text = " ".join(full_text_parts)

    all_words = full_text.split()

    # Cần map lại: từ chỉ số từ (word index) -> vị trí ký tự trong full_text để tra trang.
    # Tính vị trí ký tự bắt đầu của từng từ 1 lần duy nhất (đỡ phải join lại nhiều lần).
    word_char_starts: List[int] = []
    pos = 0
    for w in all_words:
        idx = full_text.find(w, pos)
        word_char_starts.append(idx)
        pos = idx + len(w)

    def _page_for_char(char_pos: int) -> int:
        for page_number, start, end in page_char_ranges:
            if start <= char_pos <= end:
                return page_number
        return page_char_ranges[-1][0]  # fallback: trang cuối

    def _assign_page_number(word_start_idx: int, word_end_idx: int) -> int:
        """Gán page_number theo trang chiếm đa số ký tự trong khoảng từ [start, end)."""
        chunk_words = all_words[word_start_idx:word_end_idx]
        chunk_text = " ".join(chunk_words)
        char_start = word_char_starts[word_start_idx]
        char_end = char_start + len(chunk_text)

        # Đếm số ký tự chunk này rơi vào từng trang
        counts: dict = {}
        for page_number, p_start, p_end in page_char_ranges:
            overlap_start = max(char_start, p_start)
            overlap_end = min(char_end, p_end)
            if overlap_end > overlap_start:
                counts[page_number] = counts.get(page_number, 0) + (overlap_end - overlap_start)

        if not counts:
            return _page_for_char(char_start)

        max_count = max(counts.values())
        # Nếu chênh lệch < 10% tổng ký tự -> lấy trang bắt đầu (đầu tiên theo thứ tự trang)
        total = sum(counts.values())
        top_pages = [p for p, c in counts.items() if c == max_count]
        if len(top_pages) > 1 or (total > 0 and max_count / total < 0.55):
            return _page_for_char(char_start)  # coi như "gần bằng" -> lấy trang bắt đầu
        return top_pages[0]

    # --- Chia cố định FIXED_CHUNK_WORDS từ, overlap OVERLAP_WORDS ---
    n = len(all_words)
    start = 0
    idx = 0
    while start < n:
        end = min(start + FIXED_CHUNK_WORDS, n)
        piece_words = all_words[start:end]
        piece_text = " ".join(piece_words)
        page_number = _assign_page_number(start, end)

        chunks.append(
            ChunkDict(
                chunk_id=f"{source_file}_fixed_c{idx}",
                source_file=source_file,
                page_number=page_number,
                page_range=None,
                text=piece_text,
                token_count=token_counter(piece_text),
                is_bridge=False,
            )
        )
        idx += 1

        if end >= n:
            break
        start = max(end - OVERLAP_WORDS, start + 1)

    return chunks


if __name__ == "__main__":
    # Ví dụ test KHÔNG cần tải tokenizer thật -> dùng token_counter giả (đếm theo từ)
    # để kiểm tra logic chia đoạn/overlap/bridge độc lập với model.
    def fake_token_counter(text: str) -> int:
        # Xấp xỉ ~1.5 token/từ tiếng Việt, chỉ dùng để TEST LOGIC, không dùng thật
        return int(len(text.split()) * 1.5)

    demo_pages = [
        {"page_number": 1, "source_file": "demo.pdf", "text": " ".join([f"từ{i}" for i in range(200)])},
        {"page_number": 2, "source_file": "demo.pdf", "text": " ".join([f"từ{i}" for i in range(200, 400)])},
        {"page_number": 3, "source_file": "demo.pdf", "text": " ".join([f"từ{i}" for i in range(400, 600)])},
    ]

    page_chunks = chunk_by_page(demo_pages, token_counter=fake_token_counter)
    fixed_chunks = chunk_fixed_size(demo_pages, token_counter=fake_token_counter)

    print(f"chunk_by_page: {len(page_chunks)} chunk (gồm cả bridge)")
    print(f"chunk_fixed_size: {len(fixed_chunks)} chunk")
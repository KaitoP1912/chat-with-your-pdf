# BÁO CÁO TIẾN ĐỘ THỰC TẬP TỐT NGHIỆP — TUẦN 3

**Đề tài:** Xây dựng ứng dụng hỏi đáp tài liệu PDF tiếng Việt có trích dẫn số trang bằng kỹ thuật RAG (Chat with Your PDF).
**Sinh viên thực hiện:** Võ Thành Phước — MSSV: 079205022977.
**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Tiến.
**Thời gian báo cáo:** Tuần 3 (Hoàn thiện Trạm 2 — Page-aware Chunking, Embedding & Indexing).
**Quản lý mã nguồn (GitHub):** <https://github.com/KaitoP1912/chat-with-your-pdf>

---

## 1. Mục tiêu Tuần 3

Trạm 1 đã khóa an toàn ở Tuần 2 (F1=1.000, CER=0.003, FCR=0.000, 138/138 pytest pass). Mục tiêu cốt lõi Tuần 3 là xây dựng hoàn chỉnh **Trạm 2 (Page-aware Chunking, Embedding & Indexing)**, phục vụ trực tiếp câu hỏi nghiên cứu trung tâm của đồ án:

1. Nối kết quả Trạm 1 thành luồng dữ liệu sạch (số trang + văn bản chuẩn hóa) sẵn sàng cho chunking.
2. Xây dựng 2 chiến lược chia đoạn để so sánh: page-aware chunking (có bridge chunk) và fixed-size baseline.
3. Tích hợp mô hình embedding tiếng Việt và lập chỉ mục vector FAISS.
4. Đo thực nghiệm số lượng chunk, thời gian xử lý, dung lượng RAM trên toàn bộ corpus hợp lệ.
5. Chứng minh luồng truy hồi tối thiểu chạy được: 1 câu hỏi → top-k đoạn văn bản đúng, kèm đúng số trang.

---

## 2. Công việc đã thực hiện

### 2.1. Nối dữ liệu Trạm 1 → Trạm 2 (`source/retrieval/ingest_glue.py`)

Trạm 1 gồm 2 hàm độc lập, chưa có sẵn điểm nối: `pdf_loader.load_pdf_pages()` trả về `PageData` (có `page_number`, `source_file`, nhưng `raw_text` **chưa** chuẩn hóa); `text_normalizer.normalize_page_text()` nhận `raw_text` và trả về `normalized_text` (chữ sạch) nhưng **không** giữ số trang. Hàm `build_clean_pages()` được viết để ghép 2 kết quả này thành 1 danh sách trang hoàn chỉnh `{page_number, source_file, text}` — đầu vào chuẩn cho toàn bộ Trạm 2.

### 2.2. Tách từ tiếng Việt (`source/retrieval/word_segmenter.py`)

- Dùng `py_vncorenlp`, chỉ load annotator `wseg` (word segmentation) thay vì mặc định cả 4 annotator (`wseg`, `pos`, `ner`, `parse`) — giảm thời gian khởi tạo model từ ~30 giây xuống còn **~0,75 giây**, vì Trạm 2 chỉ cần tách từ, không cần gán nhãn từ loại hay nhận diện thực thể.
- Áp dụng singleton pattern: model VnCoreNLP chỉ khởi tạo 1 lần cho toàn bộ chương trình (JVM không nên khởi tạo lại nhiều lần).
- **Lỗi thực nghiệm đã xử lý:** `py_vncorenlp` bắt buộc đường dẫn `save_dir` phải là **đường dẫn tuyệt đối**; đường dẫn tương đối gây lỗi JVM khó hiểu (`java.lang.NoClassDefFoundError: vn/pipeline/VnCoreNLP`) — đã cố định bằng `Path(__file__).resolve()` trong toàn bộ script gọi tới.
- **Lỗi dữ liệu đã xử lý:** file từ điển `models/wordsegmenter/vi-vocab` bị thiếu do cơ chế tải model của `py_vncorenlp` dùng lệnh `wget` qua `os.system()`, không tương thích với `cmd.exe` trên Windows (khác với alias `wget` có sẵn trong PowerShell) → tải lại thủ công bằng `Invoke-WebRequest`. Sau khi khắc phục, điểm tương đồng cosine của kết quả truy hồi tăng rõ rệt (0,62 → 0,73 trên cùng 1 câu hỏi kiểm thử), xác nhận việc tách từ thiếu từ điển làm giảm chất lượng embedding.

### 2.3. Hai chiến lược chunking (`source/retrieval/chunker.py`)

**a) `chunk_by_page()` — chiến lược đề xuất:**
- Chia mỗi trang thành đoạn con **≤256 token** (đếm bằng tokenizer thật của model embedding), overlap **~30 từ** giữa các đoạn liền kề trong cùng trang.
- **Bridge chunk:** tại mỗi ranh giới trang N/N+1, sinh thêm 1 đoạn gồm **128 từ cuối trang N + 128 từ đầu trang N+1**, gắn nhãn `page_range="N-N+1"`, cờ `is_bridge=True`. Chỉ cần lọt top-3 khi truy hồi, không cần định vị chính xác tuyệt đối.
- **Lỗi thực nghiệm phát hiện và xử lý:** nếu 1 trong 2 trang liền kề trống (trang scan không trích được text), bridge chunk sinh ra ban đầu bị trùng lặp y hệt nội dung chunk thường (vì chỉ lấy được từ 1 phía) — đã sửa: bridge chunk chỉ sinh ra khi **cả 2 phía đều có nội dung**, tránh gây nhiễu dữ liệu khi truy hồi.

**b) `chunk_fixed_size()` — baseline so sánh:**
- Nối toàn bộ văn bản của tài liệu (không tôn trọng ranh giới trang), chia cố định **170 từ/chunk** (cỡ tối đa PhoBERT-based mã hóa trọn vẹn, ước lượng từ giới hạn 256 token), overlap 30 từ — cùng tham số overlap với `chunk_by_page` để đảm bảo so sánh công bằng.
- Với chunk vắt qua 2 trang gốc: gán `page_number` theo trang **chiếm đa số ký tự**; nếu tỉ lệ gần bằng nhau, lấy trang bắt đầu.

### 2.4. Embedding + FAISS (`source/retrieval/vectorstore.py`)

- Model: `bkai-foundation-models/vietnamese-bi-encoder` (768 chiều, `max_seq_length=256`, backbone PhoBERT-base-v2, bắt buộc input đã tách từ).
- Encode bằng `sentence-transformers`, L2-normalize vector trước khi lập chỉ mục.
- Chỉ mục: FAISS `IndexFlatIP` (inner product trên vector đã normalize = cosine similarity), tạo mới mỗi phiên, không lưu trữ lâu dài — đúng phạm vi đồ án.
- Lưu song song mapping từ vị trí trong FAISS → metadata chunk gốc (`chunk_id`, `page_number`/`page_range`, `is_bridge`, `text`), vì FAISS chỉ trả về chỉ số nguyên.

### 2.5. Script thực nghiệm (`script/pilot_tuan3/`)

| Script | Vai trò |
|---|---|
| `demo_retrieval.py` | Chạy luồng end-to-end: 1 câu hỏi → top-k chunk kèm số trang. Tự động lưu log ra `results/tuan3_pilot/demo_retrieval_<timestamp>.txt` mỗi lần chạy. |
| `measure_tuan3_corpus.py` | Đo số chunk, thời gian, RAM ước tính trên toàn bộ corpus hợp lệ, xuất `results/tuan3_pilot/tuan3_measurement_summary.csv`. |

---

## 3. Kết quả thực nghiệm

### 3.1. Phạm vi corpus áp dụng cho Trạm 2

Trong 9 file corpus chính thức (khóa từ Tuần 1), Trạm 2 chạy trên **7 file PDF hợp lệ** — loại `scan_nd238_14tr.pdf` (đã bị Trạm 1 từ chối xử lý vì full scan) và `word_118-2025-qh15_28tr.docx` (định dạng Word, thuộc phần mở rộng, `pdf_loader`/`chunker` hiện chỉ xử lý PDF).

### 3.2. Bảng số liệu chunking trên toàn corpus

| File | Trang | chunk_by_page (thường/bridge/tổng) | chunk_fixed_size | Thời gian (s) |
|---|---|---|---|---|
| normal_hienphap_33tr.pdf | 33 | 94/32/126 | 103 | 69,29 |
| normal_vinamilkbaocao2014_53tr.pdf | 53 | 189/52/241 | 194 | 148,71 |
| normal_lichsudang_C1&2_60tr.pdf | 60 | 230/57/287 | 258 | 149,35 |
| mixedscan_qcvn06_38tr.pdf | 38 | 118/29/147 | 124 | 78,65 |
| oldenc_vni_118-2025-qh15_23tr.pdf | 23 | 91/22/113 | 108 | 58,68 |
| oldenc_tcvn3_36-2024-qh15_53tr.pdf | 53 | 215/52/267 | 260 | 116,89 |
| nodiacritic_118-2025-qh15_20tr.pdf | 20 | 113/19/132 | 108 | 57,81 |
| **Tổng** | **280** | **1050/263/1313** | **1155** | **679,38 (≈11,3 phút)** |

RAM ước tính (vector `chunk_by_page`, float32, 768 chiều): **3,85 MB** — không đáng lo cho quy mô corpus hiện tại.

Nhận xét: `chunk_fixed_size` sinh nhiều chunk hơn `chunk_by_page` (không tính bridge) ở 6/7 file, do cỡ cố định 170 từ nhỏ hơn cỡ trung bình của chunk theo trang đối với các trang có mật độ chữ cao.

### 3.3. Đối chiếu chéo bridge chunk với dữ liệu scan/trống (Tuần 2)

Số bridge chunk sinh ra tự động khớp chính xác với số trang scan/trống đã ghi nhận ở bảng Ingestion Tuần 2, xác nhận cơ chế "bỏ qua bridge nếu 1 phía trống" hoạt động đúng trên dữ liệu thật:

| File | Số ranh giới trang | Bridge kỳ vọng nếu không có trang trống | Bridge thực tế | Giải thích chênh lệch |
|---|---|---|---|---|
| normal_lichsudang_C1&2_60tr.pdf | 59 | 59 | 57 | Khớp đúng: trang 2 là trang trống duy nhất, nằm giữa tài liệu (không phải trang biên) → làm mất đúng 2 ranh giới liền kề (1–2) và (2–3) |
| mixedscan_qcvn06_38tr.pdf | 37 | 37 | 29 | Khớp đúng phép tính từ 2 trang scan (1, 38) + 3 trang trống (3, 7, 9): 8 ranh giới bị ảnh hưởng |
| 5 file còn lại | — | — | — | Không có trang trống → bridge = số ranh giới, không chênh lệch |

### 3.4. Demo truy hồi end-to-end (nghiệm thu Bước 8)

Chạy trên `normal_hienphap_33tr.pdf`, câu hỏi: *"Vai trò của Mặt trận Tổ quốc Việt Nam là gì?"*

| Hạng | Vị trí | Score (cosine) | Nội dung |
|---|---|---|---|
| #1 | Trang 3 | 0,7259 | Mặt trận Tổ quốc Việt Nam là bộ phận của hệ thống chính trị... |
| #2 | Trang 3–4 (bridge) | 0,7109 | (nối liền nội dung 2 trang, đúng chủ đề) |
| #3 | Trang 4 | 0,5534 | ...đại diện, bảo vệ quyền và lợi ích hợp pháp của Nhân dân... |

Top-3 đều đúng chủ đề câu hỏi (nội dung thực tế thuộc Điều 9 Hiến pháp — quy định về Mặt trận Tổ quốc Việt Nam), số trang trả về chính xác. Log đầy đủ lưu tại `results/tuan3_pilot/demo_retrieval_20260816_041136.txt`.

---

## 4. Tổng kết & Đối chiếu mục tiêu Tuần 3

| Hạng mục cam kết | Trạng thái | Kết quả / Minh chứng |
| --- | --- | --- |
| **Nối dữ liệu Trạm 1 → Trạm 2** | **ĐẠT** | `source/retrieval/ingest_glue.py`, kiểm chứng trên toàn bộ 7 file corpus |
| **Tách từ tiếng Việt** | **ĐẠT** | `source/retrieval/word_segmenter.py`, tối ưu thời gian khởi tạo model 30s → 0,75s |
| **Page-aware chunking + bridge chunk** | **ĐẠT** | `chunk_by_page()`, đối chiếu chéo đúng với dữ liệu scan/trống Tuần 2 |
| **Fixed-size baseline** | **ĐẠT** | `chunk_fixed_size()`, cùng tham số overlap để so sánh công bằng |
| **Embedding + FAISS** | **ĐẠT** | `vectorstore.py`, model `vietnamese-bi-encoder`, `IndexFlatIP` |
| **Đo thực nghiệm toàn corpus** | **ĐẠT** | `results/tuan3_pilot/tuan3_measurement_summary.csv`, 7/7 file chạy không lỗi |
| **Luồng truy hồi tối thiểu chạy được** | **ĐẠT** | `demo_retrieval.py`, top-3 đúng chủ đề và đúng số trang trên dữ liệu thật |

---

## 5. Vấn đề tồn đọng và kế hoạch xử lý

- **Đếm token qua HuggingFace API mỗi lần chunk:** thời gian chunking (13–34 giây/file) còn cao do thuật toán chia đoạn gọi tokenizer nhiều lần theo cơ chế mở rộng dần cửa sổ từ. Chưa ảnh hưởng tới khả năng chạy đúng, nhưng có thể tối ưu ở tuần sau nếu cần tăng tốc trên corpus lớn hơn.
- **Chưa xử lý file `.docx`:** đúng phạm vi đã thống nhất (Word là phần mở rộng), sẽ xem xét ở Tuần 7 nếu còn thời gian.
- **Chưa chọn ngưỡng cosine similarity để lọc kết quả:** cố ý để lại cho Tuần 4 (chọn ngưỡng dựa trên dev set, đúng thứ tự công việc trong đề cương).

---

## 6. Kế hoạch triển khai Tuần 4 (RAG Pipeline & Fixed-size Baseline)

Trạm 2 đã hoàn tất. Trọng tâm Tuần 4 chuyển sang xây dựng **Trạm 3 (Retrieval)** hoàn chỉnh và tích hợp **Trạm 4 (QA & Citation Generation)** ở mức cơ bản:

1. **Chọn ngưỡng cosine similarity:** dùng 10 câu hỏi dev set (đã có từ Tuần 2) để thử dải ngưỡng, chọn ngưỡng tối thiểu cho retrieval.
2. **Tích hợp Gemini cho sinh câu trả lời:** ghép kết quả top-k retrieval vào prompt, sinh câu trả lời kèm trích dẫn số trang lấy từ metadata (không cho LLM tự tạo số trang).
3. **Chạy song song 2 cấu hình (page-aware vs fixed-size)** trên cùng dev set để có số liệu Hit@3 sơ bộ đầu tiên.
4. **Đánh giá và nghiệm thu Tuần 4:** có metric trên dev set (Hit@3, answer correctness sơ bộ) theo đúng yêu cầu nghiệm thu chung của giảng viên.

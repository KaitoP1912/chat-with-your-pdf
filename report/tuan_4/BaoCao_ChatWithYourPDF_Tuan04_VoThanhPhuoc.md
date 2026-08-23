# BÁO CÁO TIẾN ĐỘ THỰC TẬP TỐT NGHIỆP — TUẦN 4

**Đề tài:** Xây dựng ứng dụng hỏi đáp tài liệu PDF tiếng Việt có trích dẫn số trang bằng kỹ thuật RAG (Chat with Your PDF).
**Sinh viên thực hiện:** Võ Thành Phước — MSSV: 079205022977.
**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Tiến.
**Thời gian báo cáo:** Tuần 4 (Hoàn thiện Trạm 3 — Retrieval calibration & Trạm 4 — QA/Citation Generation cơ bản).
**Quản lý mã nguồn (GitHub):** <https://github.com/KaitoP1912/chat-with-your-pdf>

---

## 1. Mục tiêu Tuần 4

Trạm 2 đã khóa an toàn ở Tuần 3 (1313 chunk page-aware gồm 263 bridge chunk, 1155 chunk fixed-size baseline; demo truy hồi end-to-end đúng chủ đề, đúng số trang). Mục tiêu cốt lõi Tuần 4:

1. Mở rộng dev set để có đủ câu hỏi answerable **và** unanswerable, phục vụ hiệu chỉnh ngưỡng abstention có ý nghĩa (theo góp ý của Thầy, email 16/08/2026).
2. Chọn ngưỡng cosine similarity (tau) dựa trên dev set, không chọn cảm tính — hiệu chỉnh riêng theo từng cấu hình chunking.
3. Tích hợp Gemini sinh câu trả lời kèm trích dẫn số trang lấy từ metadata (không cho LLM tự tạo số trang).
4. Chạy song song 2 cấu hình (page-aware vs fixed-size) trên cùng dev set để có số liệu Hit@3 và các chỉ số end-to-end đầu tiên.
5. Áp quy tắc quyết định (Thầy đề ra) để chốt kiến trúc chunking mặc định cho ứng dụng.

---

## 2. Công việc đã thực hiện

### 2.1. Mở rộng và chuẩn hóa Dev Set (`script/pilot_tuan4/normalize_dev_questions.py`)

Dev set mở rộng từ 25 câu (Tuần 3) lên **34 câu**, bổ sung theo đúng yêu cầu của Thầy: *"bộ câu hỏi đánh giá từ Tuần 4 phải có cả answerable và unanswerable questions, nếu không thì không thể xác định ngưỡng abstention một cách có ý nghĩa."*

- **23 câu answerable** (có đáp án thật trong tài liệu, gồm cả câu bridge — đáp án trải 2 trang liền kề).
- **11 câu unanswerable**, chia 2 nhóm:
  - **5 câu `out_of_scope`**: hỏi về chủ đề hoàn toàn không liên quan tài liệu (VD: thuế thu nhập cá nhân trong Hiến pháp, thị phần Tesla).
  - **6 câu `near_miss`**: hỏi về chủ đề *có vẻ* liên quan (cùng lĩnh vực, cùng tài liệu) nhưng chi tiết cụ thể được hỏi không hề tồn tại trong văn bản (VD: mức lương cụ thể của Tổng Giám đốc, số súng đạn cụ thể tại 1 trận đánh) — nhóm này khó hơn `out_of_scope` vì điểm cosine dễ bị đẩy lên do trùng chủ đề.

Do các câu unanswerable mới thêm thiếu field so với câu cũ, script `normalize_dev_questions.py` bổ sung mặc định hợp lý (`answer_reference`, `answer_pages`, `is_bridge_case`, `out_of_scope`, `type`) để toàn bộ 34 câu dùng chung 1 schema, xuất ra `data/eval_sets/dev_questions_normalized.json`.

### 2.2. Sửa lỗi Chunking phát hiện qua dev set mới (`source/retrieval/chunker.py`)

Trong lúc thử nghiệm retrieval trên dev set mới, phát hiện `dev_13`, `dev_14` (câu hỏi ghép 2 ý nằm trong cùng 1 trang dài, ví dụ Điều 94 + Điều 95) bị rớt điểm cosine dưới ngưỡng dù thông tin có thật trong tài liệu. Nguyên nhân: khi 1 trang dài bị `_chunk_page_text()` cắt thành ≥2 mảnh do vượt `MAX_TOKENS_PER_CHUNK`, overlap 30 từ giữa 2 mảnh liền kề là quá ít để giữ đủ tín hiệu ngữ nghĩa cho câu hỏi ghép ý.

**Khắc phục:** bổ sung **bridge chunk nội bộ trang** (128 từ mỗi bên, dùng chung logic "bỏ qua nếu 1 phía rỗng" như bridge trang-trang) tại mỗi ranh giới mảnh-mảnh trong cùng 1 trang, tách biệt với bridge trang-trang đã có từ Tuần 3. Hệ quả: số chunk `page_aware` tăng đáng kể so với Tuần 3 (VD `normal_hienphap_33tr.pdf`: 126 → **187 chunk**) — đây là do bugfix, không phải do đổi tham số `MAX_TOKENS_PER_CHUNK`/`OVERLAP_WORDS` đã chốt.

### 2.3. Nâng cấp Retrieval — Hybrid Search (`source/retrieval/vectorstore.py`)

Trước khi tích hợp chính thức, đã thử nghiệm riêng qua `test_query_decomposition_retrieval.py` (script thăm dò, không dùng số liệu báo cáo chính thức). Sau khi xác nhận hiệu quả, tích hợp **Hybrid Search (BM25 + Dense) qua Reciprocal Rank Fusion (RRF)** vào `ChunkIndex` chính thức:

- Giữ nguyên FAISS `IndexFlatIP` (dense/cosine) từ Tuần 3.
- Bổ sung BM25 (`rank-bm25`) trên cùng tập chunk đã tách từ.
- Hợp nhất thứ hạng bằng RRF (`rrf_k=60`).
- **Điểm `score` trả về cho từng hit vẫn là cosine similarity gốc** (không phải điểm RRF) — để dùng độc lập làm Dense Guardrail cho việc quyết định ngưỡng abstention ở bước sau.

Lý do bổ sung BM25: các câu hỏi tra cứu số liệu/tên riêng cụ thể (VD dev_13, dev_14) được BM25 kéo đúng chunk lên hạng cao hơn so với dense thuần.

### 2.4. Đo Retrieval thô trên toàn Dev Set (`script/pilot_tuan4/run_dev_retrieval.py`)

Chạy retrieval (k=3) cho cả 2 cấu hình chunking trên 34 câu × 2 = 68 lượt, xây riêng 2 FAISS index (page_aware, fixed_size) cho mỗi 1 trong 3 file nguồn (đúng kiến trúc "1 tài liệu/phiên" đã chốt), xuất `results/tuan4_pilot/dev_retrieval_raw.csv`.

**Kết quả Hit@3 (trên 23 câu answerable):**

| Cấu hình | Hit@3 | Câu bị miss |
|---|---|---|
| **page_aware** | **22/23 = 95.7%** | dev_17 |
| fixed_size | 21/23 = 91.3% | dev_03, dev_17 |

page_aware cao hơn fixed_size **+4.4 điểm phần trăm**.

### 2.5. Hiệu chỉnh ngưỡng Cosine (Threshold Calibration) (`script/pilot_tuan4/threshold_sweep.py`)

Quét dải ngưỡng tau từ 0.30 đến 0.80 (bước 0.05), tính đồng thời 2 vai trò của ngưỡng theo đúng chỉ đạo của Thầy (email 16/08/2026): *"Cosine threshold vừa dùng để loại evidence quá yếu, vừa là tín hiệu để quyết định 'không tìm thấy thông tin trong tài liệu'."*

Với mỗi ngưỡng, tính: false abstention rate (câu answerable bị từ chối oan), false acceptance rate (câu unanswerable bị chấp nhận sai), và Abstention Precision/Recall/F1 (coi việc "từ chối trả lời" là bài toán phân loại nhị phân, để đo riêng khả năng phát hiện câu hỏi không có thông tin — đúng góp ý bổ sung của Thầy vào rubric).

**Ngưỡng theo sweep tự động** (tiêu chí: Abstention F1 cao nhất, tie-break: false acceptance rate thấp hơn):

| Cấu hình | Tau đề xuất | Abstention F1 | False abstention rate | False acceptance rate |
|---|---|---|---|---|
| **page_aware** | **0.50** | **0.9167** | 8.7% (2/23) | **0.0%** (0/11) |
| fixed_size | 0.45 | 0.88 | 13.0% (3/23) | **0.0%** (0/11) |

Trong lúc đối chiếu số liệu, phát hiện phân bố điểm cosine của dev set có đặc điểm đáng chú ý: 2 câu answerable khó nhất (dev_13=0.3994, dev_14=0.4154 — câu hỏi ghép 2 ý trong cùng 1 trang) có điểm thấp hơn cả ngưỡng sweep 0.45–0.50, trong khi 1 số câu unanswerable (dev_20=0.4696, dev_30=0.4376, dev_33=0.4069) lại có điểm cao hơn mốc 0.38. Tức là điểm của 2 nhóm câu answerable/unanswerable bị **chồng lấn nhau** trong khoảng 0.37–0.47, không có 1 mốc "trong" nào tách bạch hoàn toàn 2 nhóm. Đây là lý do cần đánh giá **2 kịch bản ngưỡng khác triết lý**, thay vì chỉ chốt theo 1 con số sweep tự động (mục 2.6 và 3.4 dưới đây).

**Lưu ý kỹ thuật:** ban đầu module `qa_generator.py` chỉ nhận 1 tham số `tau` dùng chung cho cả 2 cấu hình, mặc định lấy `DEFAULT_TAU = 0.38`. Đã sửa `run_dev_qa.py` để có thể chạy được **cả 2 kịch bản tường minh, tách file kết quả riêng** (`--tau 0.38` cho ngưỡng chung, hoặc `--tau-page-aware`/`--tau-fixed-size` cho ngưỡng riêng theo sweep) — phục vụ so sánh trực tiếp ở mục 3.4, tránh lẫn ngưỡng giữa 2 lần chạy.

### 2.6. Tích hợp Gemini QA & Citation Generation (`source/qa/qa_generator.py`)

Xây dựng module sinh câu trả lời 2 lớp phòng vệ:

1. **Retrieval Abstention (tau):** nếu không có chunk nào đạt ngưỡng cosine đã hiệu chỉnh riêng theo cấu hình → từ chối ngay, không gọi API.
2. **Model Refusal:** nếu retrieval qua ngưỡng nhưng bản thân Gemini đánh giá ngữ cảnh không đủ để trả lời chắc chắn → bắt buộc trả về đúng câu *"Không tìm thấy thông tin trong tài liệu."* theo prompt đã ràng buộc.

Citation lấy trực tiếp từ metadata chunk (`page_number`/`page_range`, `is_bridge`) đưa vào prompt — **không để LLM tự sinh số trang**, đúng yêu cầu đã thống nhất với Thầy.

Prompt có Guardrail chống prompt injection (kế thừa từ baseline Tuần 1): yêu cầu model bỏ qua mọi chỉ dẫn xuất hiện bên trong nội dung tài liệu trích dẫn.

Khóa cố định 1 model duy nhất `gemini-3.5-flash-lite` cho toàn phiên chạy (có cơ chế cảnh báo nếu phát hiện nhiều model khác nhau được dùng lẫn trong cùng dev set, xảy ra ở 2 lần chạy thử ban đầu do model cũ `gemini-2.5-flash` bị deprecated giữa chừng — đã khắc phục bằng cách liệt kê lại danh sách model khả dụng và cố định `gemini-3.5-flash-lite`).

### 2.7. Chạy QA trên toàn Dev Set theo 2 kịch bản ngưỡng song song (`script/pilot_tuan4/run_dev_qa.py`)

Vì phân bố điểm cosine của 2 nhóm câu answerable/unanswerable chồng lấn nhau (mục 2.5), quyết định chạy đầy đủ **2 kịch bản ngưỡng có triết lý khác nhau**, để có số liệu thật đối chiếu thay vì chỉ tranh luận lý thuyết:

- **Kịch bản A — Ngưỡng riêng theo sweep** (`page_aware=0.5`, `fixed_size=0.45`): chặn thẳng ở tầng Retrieval, ngưỡng số cứng, không phụ thuộc hành vi model.
- **Kịch bản B — Ngưỡng chung thấp hơn** (`tau=0.38` cho cả 2 cấu hình): giữ đúng 1 biến số duy nhất (cách chunking) khi so sánh 2 cấu hình — nguyên tắc thực nghiệm có kiểm soát (controlled experiment) thường dùng khi so sánh 2 phương pháp. Đổi lại, ngưỡng thấp hơn để lọt nhiều câu unanswerable qua tầng 1, phải trông cậy vào tầng 2 (Model Refusal — Gemini tự nhận biết ngữ cảnh không đủ) chặn tiếp.

Cả 2 kịch bản chạy Gemini QA đầy đủ 34 câu × 2 cấu hình = 68 lượt (**136 lượt gọi API tổng cộng cho cả 2 kịch bản**), đo latency, token, tự động kiểm citation accuracy, và chấm tay `answer_correctness_manual` (đúng hoàn toàn / đúng một phần / sai) cho từng câu trả lời được. Kết quả 68/68 dòng thành công ở cả 2 kịch bản, 0 lỗi API, thống nhất 1 model `gemini-3.5-flash-lite`.

---

## 3. Kết quả thực nghiệm tổng hợp Bước 5

### 3.1. Hit@3 (không phụ thuộc tau, đo 1 lần ở Bước 2)

| Cấu hình | Hit@3 |
|---|---|
| **page_aware** | **22/23 = 95.7%** |
| fixed_size | 21/23 = 91.3% |

### 3.2. So sánh đầy đủ 2 kịch bản ngưỡng (Bước 4-5)

| Chỉ số | **Kịch bản A** — page_aware (tau=0.5) | **Kịch bản A** — fixed_size (tau=0.45) | **Kịch bản B** — page_aware (tau=0.38) | **Kịch bản B** — fixed_size (tau=0.38) |
|---|---|---|---|---|
| Answerable trả lời được | 16/23 | 17/23 | **21/23** | **21/23** |
| Answer correctness (đúng hoàn toàn / một phần / **sai**) | 11 / 5 / **0** | 12 / 5 / **0** | 14 / 6 / **1** | 13 / 7 / **1** |
| Citation accuracy | 15/16 = 93.8% | 15/17 = 88.2% | 20/21 = 95.2% | 19/21 = 90.5% |
| Unanswerable: chặn ở retrieval / chặn ở model / **lọt hẳn** | 11 / 0 / **0** | 11 / 0 / **0** | 7 / 4 / **0** | 8 / 3 / **0** |
| Abstention F1 (theo sweep, riêng tầng retrieval) | 0.9167 | 0.88 | *(không áp dụng — cùng tau=0.38 cả 2 nhóm, xem phân tích 3.3)* | |
| Latency trung bình | 1.05s | 1.01s | 1.07s | 1.01s |
| Token trung bình/câu | 1282 | 1041 | 1368 | 1139 |

### 3.3. Phát hiện quan trọng: đánh đổi thật giữa 2 kịch bản

**Điểm mạnh của Kịch bản B:** trả lời được nhiều câu hơn hẳn (21/23 so với 16-17/23), và **0/11 câu unanswerable "lọt hẳn"** ở cả 2 cấu hình — tức dù có 7-8 câu unanswerable vượt qua tầng Retrieval (do ngưỡng thấp), toàn bộ đều bị tầng Model Refusal (Gemini tự nhận biết không đủ thông tin) chặn lại thành công trong lần chạy này.

**Rủi ro cụ thể đã quan sát được ở Kịch bản B:** câu `dev_16` (hỏi tỷ lệ sở hữu cổ phần cụ thể của Vinamilk) bị chấm **"sai"** ở cả 2 cấu hình — mô hình tự đưa ra 1 con số (37,9%) không có trong tài liệu, trong khi ở Kịch bản A (ngưỡng cao hơn), cùng câu hỏi này chỉ bị chấm "đúng một phần" (không tự bịa số, chỉ trả lời thiếu). Nguyên nhân nhiều khả năng: ngưỡng thấp hơn cho phép nhiều chunk có độ liên quan thấp hơn lọt vào ngữ cảnh, làm tăng nguy cơ model suy diễn/bịa khi ngữ cảnh không đủ rõ ràng — đây là bằng chứng thực nghiệm cụ thể, không phải suy đoán lý thuyết.

**Tóm tắt đánh đổi:**

| | Kịch bản A (tau riêng, cao hơn) | Kịch bản B (tau=0.38 chung) |
|---|---|---|
| Số câu trả lời được | Ít hơn | Nhiều hơn (+5 câu mỗi cấu hình) |
| An toàn trước câu hỏi ngoài phạm vi | Chặn ngay ở tầng 1 (ngưỡng số, ổn định) | Phụ thuộc tầng 2 (hành vi model, đã đúng trong lần chạy này nhưng chưa có gì đảm bảo cho câu hỏi khác trong tương lai) |
| Rủi ro bịa thông tin trong câu trả lời | Không quan sát thấy (0/33 sai) | Đã quan sát 1 trường hợp cụ thể (dev_16, cả 2 cấu hình) |
| Tính "1 biến số" khi so sánh 2 chunking | Có 2 biến (tau khác nhau + cách chunking) | Đúng 1 biến duy nhất (chỉ cách chunking) |

### 3.4. Áp quy tắc quyết định khóa kiến trúc chunking

Theo hướng dẫn của Thầy (email 16/08/2026): *"Nếu page-aware tương đương hoặc chỉ thấp hơn cấu hình tốt nhất trong khoảng ≤3 điểm phần trăm Hit@3, giữ page-aware."*

Kết quả Hit@3 (mục 3.1, không phụ thuộc tau — đúng phần so sánh khoa học thuần túy giữa 2 cách chunking): **page_aware cao hơn fixed_size 4.4 điểm phần trăm** — vượt điều kiện ≤3pp. page_aware cũng thắng citation accuracy ở cả 2 kịch bản ngưỡng.

→ **Quyết định (không phụ thuộc việc chọn kịch bản tau nào): khóa `page-aware chunking` làm kiến trúc mặc định chính thức.** `fixed_size` giữ lại trong mã nguồn làm baseline đối chứng.

**Việc còn lại cần Thầy cho ý kiến:** chọn ngưỡng tau nào (Kịch bản A hay B) cho pipeline `page-aware` sẽ khóa chính thức ở Tuần 5 — xem mục "Nội dung cần hỏi ý kiến giảng viên" trong email báo cáo kèm theo.

---

## 4. Đối chiếu với mục tiêu Tuần 4

| Hạng mục cam kết | Trạng thái | Kết quả / Minh chứng |
| --- | --- | --- |
| **Dev set mở rộng (answerable + unanswerable)** | **ĐẠT** | 34 câu (23 answerable / 11 unanswerable: 5 out_of_scope + 6 near_miss), `dev_questions_normalized.json` |
| **Chọn ngưỡng cosine dựa trên dev set, không cảm tính** | **ĐẠT** | `threshold_sweep.py` quét 11 mốc tau × 2 cấu hình; do phân bố điểm 2 nhóm chồng lấn, dựng thêm 2 kịch bản đối chứng đầy đủ dữ liệu (mục 3.2-3.3) thay vì chỉ chốt 1 số theo sweep tự động |
| **Bổ sung Abstention Precision/Recall/F1 vào rubric** | **ĐẠT** | Số liệu tại mục 3.1, diễn giải tại mục 3.2 |
| **Tích hợp Gemini sinh câu trả lời kèm trích dẫn trang từ metadata** | **ĐẠT** | `source/qa/qa_generator.py`, citation không do LLM tự tạo |
| **Chạy song song 2 cấu hình, đo đầy đủ Hit@3 + answer correctness + citation + abstention + latency + token** | **ĐẠT** | 68/68 dòng thành công, bảng tổng hợp mục 3.1 |
| **Áp quy tắc ≤3pp để quyết định khóa kiến trúc** | **ĐẠT** | page_aware vượt fixed_size +4.4pp → khóa page-aware |

---

## 5. Vấn đề tồn đọng và kế hoạch xử lý

- **Chưa chốt ngưỡng tau cuối cùng cho pipeline production:** đang có 2 kịch bản với đánh đổi rõ ràng (mục 3.3), chưa đủ căn cứ để tự chọn 1 bên — cần ý kiến Thầy trước khi khóa ở Tuần 5 (xem email báo cáo kèm theo).
- **Model over-refusal (quan sát ở cả 2 kịch bản):** nhiều câu answerable bị chính Gemini từ chối dù retrieval đã cung cấp đúng ngữ cảnh trong top-3. *Hướng xử lý:* xem xét điều chỉnh lại prompt QA ở Tuần 5 (nới lỏng điều kiện "đủ thông tin" trong hướng dẫn model, hoặc thử nghiệm nhiệt độ/temperature) để giảm tỉ lệ này mà không làm tăng false acceptance.
- **Rủi ro bịa thông tin khi ngưỡng thấp (dev_16, Kịch bản B):** cần thêm câu hỏi tương tự (hỏi số liệu cụ thể, dễ gây suy diễn) vào test set chính thức ở Tuần 5 để đánh giá đầy đủ hơn mức độ phổ biến của rủi ro này, tránh kết luận vội từ 1 trường hợp quan sát được.
- **Model API không ổn định giữa các phiên chạy:** phát hiện `gemini-2.5-flash` bị Google ngừng hỗ trợ cho người dùng mới ngay trong tuần triển khai. *Đã xử lý:* khóa cố định `gemini-3.5-flash-lite` cho toàn bộ phiên chạy chính thức, có cảnh báo tự động trong script nếu phát hiện lẫn model.
- **Chunk count tăng đáng kể do bugfix bridge chunk nội bộ trang:** cần đo lại RAM/thời gian embedding trên toàn corpus 7 file (giống bảng đã làm ở Tuần 3) để cập nhật số liệu vận hành, vì Tuần 3 đo trên phiên bản chunker chưa có bridge nội bộ trang.
- **Chưa đo trên 2 file corpus còn lại (mixed-scan, oldenc):** dev set hiện chỉ trải trên 3 file "chuẩn" (Hiến pháp, Vinamilk, Lịch sử Đảng). *Kế hoạch:* nếu còn thời gian ở Tuần 6-7, bổ sung câu hỏi cho các file lỗi mã hóa/scan để kiểm tra độ bền của pipeline QA trên dữ liệu nhiễu.

---

## 6. Kế hoạch triển khai Tuần 5 (Hệ thống end-to-end & Khóa Test Set)

Trạm 3 (Retrieval, đã calibrate) và Trạm 4 (QA cơ bản) đã hoàn tất ở mức dev set. Theo khung tiến độ 8 tuần đã thống nhất với Thầy, Tuần 5 là mốc bắt buộc **hệ thống end-to-end phải hoạt động** và **khóa model, prompt, threshold và test set**:

1. **Baseline Long-Context:** hoàn thiện `test_longcontext_baseline.py` (đã có baseline sơ bộ từ Tuần 1) thành cấu hình đối chứng thứ 3 đầy đủ, dùng chung dev set và cùng điều kiện chạy (model, temperature) để so sánh công bằng với RAG.
2. **Khóa toàn bộ tham số hệ thống:** model QA (`gemini-3.5-flash-lite`), tau=0.5 (page-aware, kiến trúc mặc định), prompt QA — không thay đổi sau mốc này.
3. **Khóa Test Set:** soạn bộ câu hỏi test độc lập với dev set (không dùng để chọn tham số), theo đúng cấu trúc answerable/unanswerable đã áp dụng cho dev set.
4. **Giao diện tối thiểu:** kết nối luồng end-to-end (upload PDF → hỏi đáp → trả lời kèm trích dẫn trang) thành 1 pipeline chạy được, chuẩn bị cho MVP giao diện ở Tuần 6.
5. **Đánh giá và nghiệm thu Tuần 5:** xác nhận toàn bộ 3 cấu hình (page-aware, fixed-size, long-context) chạy được trên cùng test set đã khóa, sẵn sàng cho chạy đánh giá chính thức ở Tuần 7.

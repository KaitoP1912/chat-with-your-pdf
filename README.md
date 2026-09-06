# Chat With Your PDF

Ứng dụng hỏi-đáp tài liệu PDF tiếng Việt có trích dẫn số trang, sử dụng kỹ thuật RAG (Retrieval-Augmented Generation).
Đồ án thực tập tốt nghiệp — Sinh viên: **Võ Thành Phước** (MSSV: 079205022977).
Giảng viên hướng dẫn: **ThS. Nguyễn Thanh Tiến**.

---

## Kiến trúc hệ thống

Hệ thống được thiết kế theo luồng xử lý RAG đa tầng:
1. **Trạm 1 (Ingestion & Normalization):** Đọc PDF theo từng trang, phân loại Scan/Text, chuẩn hóa bảng mã cũ (TCVN3, VNI) về Unicode NFC, phát hiện mất dấu, và lọc rác header/footer.
2. **Trạm 2 (Page-aware Chunking & Indexing):** Chia đoạn văn bản giữ nguyên metadata số trang gốc (kèm bridge chunk tại ranh giới trang và nội bộ trang), nhúng vector đặc trưng bằng `vietnamese-bi-encoder` và lập chỉ mục FAISS.
3. **Trạm 3 (Retrieval):** Truy xuất ngữ nghĩa top-k đoạn văn bản liên quan nhất theo câu hỏi, kết hợp Dense (FAISS cosine) và BM25 qua Reciprocal Rank Fusion (Hybrid Search).
4. **Trạm 4 (QA & Citation Generation):** Gọi Gemini sinh câu trả lời kèm trích dẫn chính xác số trang tài liệu nguồn (từ metadata, không để model tự bịa), có cơ chế 2 lớp phòng vệ abstention (ngưỡng cosine + model tự nhận biết) và gửi kèm ảnh trang cho các trang biểu đồ/bảng phức tạp.

**Tham số hệ thống đã khóa chính thức (Tuần 4-5, xem `config.py`):** model `gemini-3.5-flash-lite`, ngưỡng tau=0.38 (dùng chung mọi cấu hình), top-k tầng generation=15, kích thước chunk tối đa=320 token, kiến trúc mặc định `page_aware`.

---

## Cấu trúc thư mục

```text
chat-with-your-pdf/
├── config.py                  # Tham số hệ thống đã khóa (Tuần 5), mọi script đọc chung từ đây
├── data/
│   ├── corpus/                # Tập tài liệu PDF kiểm thử chính thức (9 file, khóa từ Tuần 1)
│   └── eval_sets/             # dev_questions_normalized.json (34 câu, Dev Set)
│                               # test_questions.json (25 câu, Test Set độc lập, dùng cho Tuần 7)
├── source/
│   ├── ingestion/              # Trạm 1: pdf_loader, scan_detector, text_normalizer
│   ├── retrieval/               # Trạm 2 & 3: ingest_glue, word_segmenter, chunker (chunk=320),
│   │                            # vectorstore (Hybrid Dense+BM25 RRF)
│   ├── qa/                     # Trạm 4: qa_generator (Gemini QA, citation, abstention 2 tầng,
│   │                            # gửi ảnh trang biểu đồ, prompt đã nới giảm over-refusal)
│   ├── evaluation/              # Module đo Hit@k, CER, độ chính xác trích dẫn
│   └── vietnamese_wordlist_external.txt
├── script/
│   ├── app_cli.py               # Tuần 5: CLI end-to-end demo (upload PDF -> hỏi đáp -> trích dẫn)
│   ├── pilot_tuan1/             # Kịch bản thực nghiệm đo đạc Tuần 1 (gồm baseline long-context sơ bộ)
│   ├── pilot_tuan2/             # Kịch bản kiểm thử Ingestion, Encoding, Wordseg Tuần 2
│   ├── pilot_tuan3/             # Kịch bản demo truy hồi và đo đạc Chunking/Embedding/FAISS Tuần 3
│   ├── pilot_tuan4/             # Kịch bản Tuần 4: dev set, threshold, Gemini QA, chẩn đoán lỗi
│   │   ├── normalize_dev_questions.py
│   │   ├── run_dev_retrieval.py
│   │   ├── threshold_sweep.py
│   │   ├── run_dev_qa.py         # Hỗ trợ --tau chung/riêng, mặc định --k 15
│   │   ├── run_tuan4_pipeline.py
│   │   ├── check_tuan4_status.py
│   │   ├── oracle_context_test.py    # Chẩn đoán lỗi retrieval/chunking vs lỗi model
│   │   ├── run_fixes_verification.py # Kiểm chứng tăng chunk + gửi ảnh trang biểu đồ
│   │   └── archive/               # Script thử nghiệm sơ bộ, không dùng số liệu chính thức
│   └── pilot_tuan5/             # Kịch bản Tuần 5
│       ├── run_longcontext_baseline.py  # Baseline Long-Context chính thức (cấu hình đối chứng thứ 3)
│       ├── run_dev_qa_v2.py             # Chạy lại QA (page_aware) với qa_generator.py đã sửa
│       ├── grade_answers_with_gemini.py # Chấm answer_correctness_manual bằng Gemini + PDF gốc
│       └── archive/                     # File phụ trợ soạn Test Set (không phải deliverable chính)
├── tests/
│   └── test_encoding_pipeline.py  # Hệ thống kiểm thử hồi quy tự động (138 tests)
├── vncorenlp_models/           # Model VnCoreNLP (tải qua py_vncorenlp.download_model())
├── results/
│   ├── full_text_inspect/     # Văn bản trích xuất toàn văn để kiểm tra thủ công
│   ├── tuan1_pilot/           # Kết quả đo thời gian embedding và baseline Tuần 1
│   ├── tuan2_pilot/           # Kết quả tổng hợp Ingestion CSV và log tách từ Tuần 2
│   ├── tuan3_pilot/           # CSV đo chunking/embedding/FAISS và log demo truy hồi Tuần 3
│   ├── tuan4_pilot/           # CSV retrieval, threshold, kết quả Gemini QA, oracle test, thử nghiệm sửa lỗi
│   │   └── archive/            # Kết quả thử nghiệm heuristic đã loại bỏ (không dùng)
│   └── tuan5_pilot/           # CSV baseline long-context, QA v2 (page_aware đã sửa) đã chấm điểm
└── report/                    # Báo cáo tiến độ và tài liệu chốt kỹ thuật theo tuần
    ├── tuan_1/                 # report/tuan_1/chot_cau_hoi_nghien_cuu_tuan1.md
    ├── tuan_2/                 # report/tuan_2/BaoCao_ChatWithYourPDF_Tuan02_VoThanhPhuoc.md
    ├── tuan_3/                 # report/tuan_3/BaoCao_ChatWithYourPDF_Tuan03_VoThanhPhuoc.md
    ├── tuan_4/                 # BaoCao_ChatWithYourPDF_Tuan04_VoThanhPhuoc.md
    │                            # + BaoCao_BoSung_Tuan4.md (oracle test, khóa k=15/chunk=320)
    └── tuan_5/                 # BaoCao_ChatWithYourPDF_Tuan05_VoThanhPhuoc.md
```

---

## Hướng dẫn cài đặt & Chạy kiểm thử

### 1. Chuẩn bị môi trường

Yêu cầu: Python 3.10+, Java JDK (phục vụ VnCoreNLP).

```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

Kiểm tra Java JDK đã cài đúng và biến môi trường `JAVA_HOME` đã trỏ đúng chưa (VnCoreNLP chạy trên JVM nên nếu thiếu bước này, Trạm 2 sẽ báo lỗi khi khởi tạo `py_vncorenlp.VnCoreNLP`):

```powershell
java -version
```

### 1.5. Tải model VnCoreNLP (bắt buộc trước khi chạy Trạm 2 trở đi)

Thư mục `vncorenlp_models/` **không được đính kèm trong repo** (do dung lượng lớn, đã liệt kê trong `.gitignore`) và cần được tải về máy trước khi chạy bất kỳ script nào liên quan đến tách từ tiếng Việt. Chạy đoạn Python sau **một lần duy nhất**, ngay tại thư mục gốc dự án, sau khi đã kích hoạt venv và cài `requirements.txt`:

```powershell
python -c "import py_vncorenlp, os; py_vncorenlp.download_model(save_dir=os.path.abspath('vncorenlp_models'))"
```

Lệnh này sẽ tự tạo thư mục `vncorenlp_models/` và tải về đầy đủ file `.jar` cùng các model cần thiết (wordsegmenter, pos, ner, parse).

**Lưu ý riêng cho Windows:** `py_vncorenlp.download_model()` đôi khi bị lỗi thiếu file `vi-vocab` trong quá trình tải. Nếu sau khi chạy lệnh trên mà thư mục `vncorenlp_models\models\wordsegmenter\` không có file `vi-vocab`, hãy mở PowerShell tại thư mục gốc dự án và chạy tiếp:

```powershell
# 1. Tạo thư mục đích (cờ -Force giúp không bị lỗi nếu thư mục đã tồn tại)
New-Item -ItemType Directory -Path "vncorenlp_models\models\wordsegmenter" -Force

# 2. Tải file vi-vocab về đúng vị trí
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab" -OutFile "vncorenlp_models\models\wordsegmenter\vi-vocab"
```

**Kiểm tra đã tải đúng chưa:**

```powershell
python -c "import py_vncorenlp, os; s = py_vncorenlp.VnCoreNLP(save_dir=os.path.abspath('vncorenlp_models'), annotators=['wseg']); print(s.word_segment('Việt Nam là một quốc gia Đông Nam Á'))"
```

Nếu lệnh chạy không lỗi và in ra kết quả tách từ, model đã sẵn sàng để tiếp tục các bước bên dưới.

### 1.6. Cấu hình biến môi trường (`.env`)

Các script gọi Gemini API (Trạm 4, baseline long-context Tuần 1 và Tuần 5) cần một file `.env` ở thư mục gốc dự án (file này **không** commit lên git):

```powershell
# Tạo file .env ở thư mục gốc, nội dung như sau:
GEMINI_API_KEY=dán_API_key_của_bạn_vào_đây
```

Lấy API key miễn phí tại [Google AI Studio](https://aistudio.google.com/app/apikey). Nếu chưa tạo `.env` hoặc key còn để mặc định, các script sẽ báo lỗi `❌ Chưa có GEMINI_API_KEY hợp lệ trong .env.` và dừng ngay từ đầu thay vì chạy lỗi giữa chừng.

**Lưu ý phân biệt tên file:** `script/pilot_tuan1/test_longcontext_baseline*.py` là bản pilot sơ bộ Tuần 1 (chạy tay 1 câu). Bản Baseline Long-Context **chính thức** dùng cho báo cáo (chạy đủ dev set, đã sửa lỗi encoding + regex trích trang) là `script/pilot_tuan5/run_longcontext_baseline.py` — dùng đúng file này khi cần tái tạo số liệu Tuần 5.

### 2-4. Danh mục Script Tuần 1-4

Xem chi tiết lệnh chạy Trạm 1 (Tuần 2), Trạm 2 (Tuần 3), Trạm 3-4 cơ bản (Tuần 4 — dev set, threshold sweep, Gemini QA 2 kịch bản) ở các phiên bản trước của tài liệu này, không đổi. Riêng Tuần 4 bổ sung thêm 2 công cụ chẩn đoán/khắc phục lỗi sau khi có góp ý của Thầy:

#### Chẩn đoán nguyên nhân gốc (Oracle-Context Test)

Xác định 1 câu trả lời sai là do retrieval/chunking hay do model, bằng cách đưa thẳng đúng trang có đáp án vào (bỏ qua tìm kiếm tự động):

```powershell
python script/pilot_tuan4/oracle_context_test.py --dev-questions data/eval_sets/dev_questions_normalized.json --corpus-dir data/corpus --output results/tuan4_pilot/oracle_context_results.csv
```

#### Kiểm chứng phương án khắc phục (tăng chunk + gửi ảnh trang biểu đồ)

```powershell
python script/pilot_tuan4/run_fixes_verification.py --output results/tuan4_pilot/fixes_verification_v2.csv
```

**Tham số đã khóa cuối cùng sau kiểm chứng:** `MAX_TOKENS_PER_CHUNK=320` (trong `chunker.py`), `k=15` (mặc định `run_dev_qa.py`).

### 5. Danh mục Script và Lệnh chạy kiểm thử Tuần 5

Tuần 5 khóa toàn bộ tham số hệ thống, bổ sung Baseline Long-Context, sửa 2 giới hạn tồn đọng của `qa_generator.py`, soạn Test Set độc lập và kết nối luồng end-to-end.

#### a) Xem lại toàn bộ tham số đã khóa

```powershell
python config.py
```

#### b) Chạy Baseline Long-Context (cấu hình đối chứng thứ 3)

```powershell
python script/pilot_tuan5/run_longcontext_baseline.py --limit 3
# Sau khi xác nhận ổn, chạy full 34 câu:
python script/pilot_tuan5/run_longcontext_baseline.py --limit 0 --sleep 5.0 --out results/tuan5_pilot/dev_qa_results_longcontext.csv
```

#### c) Chạy lại QA (page_aware) với qa_generator.py đã sửa (gửi ảnh trang biểu đồ + giảm over-refusal)

```powershell
python script/pilot_tuan5/run_dev_qa_v2.py --limit 5
# Sau khi xác nhận ổn, chạy full:
python script/pilot_tuan5/run_dev_qa_v2.py --limit 0 --out results/tuan5_pilot/dev_qa_results_v2_page_aware.csv
```

Script tự động in cảnh báo nếu phát hiện false acceptance tăng lên khỏi 0/11 (thành tích quan trọng nhất cần giữ vững từ Tuần 4).

#### d) Chấm điểm answer_correctness_manual bằng Gemini (có kiểm chứng chéo)

```powershell
python script/pilot_tuan5/grade_answers_with_gemini.py --results results/tuan5_pilot/dev_qa_results_v2_page_aware.csv --out results/tuan5_pilot/dev_qa_results_v2_page_aware_graded.csv
```

**Lưu ý quan trọng:** đây là chấm tự động, cần đọc lướt `answer_correctness_reason` và soát lại bằng mắt các câu chấm "sai"/"một phần" trước khi dùng cho báo cáo chính thức — bằng chứng gốc dùng để chấm được mở rộng thêm 1 trang mỗi bên so với `expected_page` (tránh lỗi thiếu trang bằng chứng đã từng gặp ở `dev_10`, `dev_19`).

#### e) Chạy demo end-to-end (CLI)

```powershell
python script/app_cli.py --pdf data/corpus/normal_hienphap_33tr.pdf --question "Nhiệm kỳ Quốc hội là bao nhiêu năm?"
```

Tham số `tau`, `k`, chiến lược chunking đã khóa cứng trong code, có thể override để debug nhưng không nên đổi khi demo nghiệm thu.

**Lưu ý chung Tuần 4-5:** các lệnh gọi Gemini API thật (mục c/d ở Tuần 4, mục b/c/d ở Tuần 5) cần `GEMINI_API_KEY` hợp lệ trong `.env` và có thể phát sinh chi phí/giới hạn quota theo tier tài khoản đang dùng — luôn dry-run (`--limit` nhỏ) trước khi chạy full.

---

## Trạng thái tiến độ

* [x] **Tuần 1:** Chốt câu hỏi nghiên cứu, thiết lập Corpus 9 file, đo thực nghiệm giới hạn an toàn 60 trang / 20 MB.
* [x] **Tuần 2:** Hoàn thiện Trạm 1 (PDF Ingestion, Scan Detector, Text Normalizer đạt F1 = 1.000, CER = 0.003, FCR = 0.000 trên 67 mẫu benchmark độc lập và 166 trang corpus sạch; 138/138 pytest passed), đo thời gian tách từ (0.0154s/trang), hoàn thành 10 câu dev set.
* [x] **Tuần 3:** Hoàn thiện Trạm 2 (page-aware chunking với bridge chunk, fixed-size baseline, embedding `vietnamese-bi-encoder`, FAISS `IndexFlatIP`); chạy thành công trên 7/7 file corpus hợp lệ; demo truy hồi end-to-end top-3 đúng chủ đề và đúng số trang.
* [x] **Tuần 4:** Hoàn thiện Trạm 3 (Hybrid Search BM25+Dense qua RRF) và Trạm 4 cơ bản; mở rộng dev set lên 34 câu; khóa Kịch bản B (tau=0.38 dùng chung). **Bổ sung sau góp ý của Thầy:** Oracle-Context Test xác nhận 8/8 câu nghi vấn là lỗi retrieval/chunking (không phải lỗi model/dev set); thử tăng k=5 thất bại, chốt chính thức **k=15 + chunk=320** — cải thiện rõ, không đánh đổi false acceptance (giữ 0/11).
* [x] **Tuần 5:** Khóa toàn bộ tham số hệ thống vào `config.py`; hoàn thiện Baseline Long-Context (cấu hình đối chứng thứ 3, đúng hoàn toàn 95.7% nhưng citation accuracy chỉ 82.6%, tốn ~9 lần token so với page_aware); tích hợp gửi ảnh trang biểu đồ + giảm model over-refusal vào `qa_generator.py` chính thức (model_refusal 2/23 → 0/23, false acceptance giữ 0/11); soạn Test Set độc lập 25 câu (`test_questions.json`); kết nối luồng end-to-end (`app_cli.py`), đã chạy thử thật thành công. **Quyết định khóa kiến trúc: giữ `page_aware`** (citation accuracy ~100%, chi phí thấp hơn nhiều so với long-context).
* [ ] **Tuần 6:** Xây giao diện tối thiểu (MVP) bọc quanh pipeline hiện có, kiểm thử trên toàn bộ 9 file corpus, chuẩn bị khóa cứng hệ thống trước khi chạy Test Set chính thức ở Tuần 7.
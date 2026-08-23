# Chat With Your PDF

Ứng dụng hỏi-đáp tài liệu PDF tiếng Việt có trích dẫn số trang, sử dụng kỹ thuật RAG (Retrieval-Augmented Generation).
Đồ án thực tập tốt nghiệp — Sinh viên: **Võ Thành Phước** (MSSV: 079205022977).
Giảng viên hướng dẫn: **ThS. Nguyễn Thanh Tiến**.

---

## Kiến trúc hệ thống

Hệ thống được thiết kế theo luồng xử lý RAG đa tầng:
1. **Trạm 1 (Ingestion & Normalization):** Đọc PDF theo từng trang, phân loại Scan/Text, chuẩn hóa bảng mã cũ (TCVN3, VNI) về Unicode NFC, phát hiện mất dấu, và lọc rác header/footer.
2. **Trạm 2 (Page-aware Chunking & Indexing):** Chia đoạn văn bản giữ nguyên metadata số trang gốc (kèm bridge chunk tại ranh giới trang), nhúng vector đặc trưng bằng `vietnamese-bi-encoder` và lập chỉ mục FAISS.
3. **Trạm 3 (Retrieval):** Truy xuất ngữ nghĩa top-k đoạn văn bản liên quan nhất theo câu hỏi bằng cosine similarity trên vector đã chuẩn hóa (FAISS `IndexFlatIP`).
4. **Trạm 4 (QA & Citation Generation):** Gọi mô hình ngôn ngữ lớn (LLM) sinh câu trả lời kèm trích dẫn chính xác số trang tài liệu nguồn.

---

## Cấu trúc thư mục

```text
chat-with-your-pdf/
├── data/
│   ├── corpus/               # Tập tài liệu PDF kiểm thử chính thức (Tuần 1 & 2)
│   └── eval_sets/            # Bộ câu hỏi đánh giá (dev_questions.json)
├── source/
│   ├── ingestion/            # Trạm 1: pdf_loader, scan_detector, text_normalizer
│   ├── retrieval/            # Trạm 2 & 3: ingest_glue, word_segmenter, chunker, vectorstore
│   ├── qa/                   # Trạm 4: generator, prompt template, citation parser
│   ├── evaluation/           # Module đo Hit@k, CER, độ chính xác trích dẫn
│   └── vietnamese_wordlist_external.txt # Từ điển 10.622 từ phục vụ scoring
├── script/
│   ├── pilot_tuan1/          # Kịch bản thực nghiệm đo đạc Tuần 1
│   ├── pilot_tuan2/          # Kịch bản kiểm thử Ingestion, Encoding, Wordseg Tuần 2
│   ├── pilot_tuan3/          # Kịch bản demo truy hồi và đo đạc Chunking/Embedding/FAISS Tuần 3
│   └── ground_truth_manual_verified.csv # Bộ Benchmark kiểm chuẩn độc lập 67 mẫu
├── tests/
│   └── test_encoding_pipeline.py # Hệ thống kiểm thử hồi quy tự động (138 tests)
├── vncorenlp_models/          # Model VnCoreNLP (tải qua py_vncorenlp.download_model())
├── results/
│   ├── full_text_inspect/    # Văn bản trích xuất toàn văn để kiểm tra thủ công
│   ├── tuan1_pilot/          # Kết quả đo thời gian embedding và baseline Tuần 1
│   ├── tuan2_pilot/          # Kết quả tổng hợp Ingestion CSV và log tách từ Tuần 2
│   └── tuan3_pilot/          # CSV đo chunking/embedding/FAISS và log demo truy hồi Tuần 3
└── report/                   # Báo cáo tiến độ và tài liệu chốt kỹ thuật theo tuần
    ├── tuan_1/                # report/tuan_1/chot_cau_hoi_nghien_cuu_tuan1.md
    ├── tuan_2/                # report/tuan_2/BaoCao_ChatWithYourPDF_Tuan02_VoThanhPhuoc.md
    ├── tuan_3/                # report/tuan_3/BaoCao_ChatWithYourPDF_Tuan03_VoThanhPhuoc.md
    └── tuan_4/                # (sẵn sàng tiếp nhận các báo cáo cho Tuần 4)
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

Các script ở Trạm 4 (QA & Citation Generation) và baseline long-context (`test_longcontext_baseline.py`, `test_longcontext_baseline_114tr.py`) cần gọi Gemini API, do đó cần một file `.env` ở thư mục gốc dự án (file này **không** commit lên git):

```powershell
# Tạo file .env ở thư mục gốc, nội dung như sau:
GEMINI_API_KEY=dán_API_key_của_bạn_vào_đây
```

Lấy API key miễn phí tại [Google AI Studio](https://aistudio.google.com/app/apikey). Nếu chưa tạo `.env` hoặc key còn để mặc định, các script sẽ báo lỗi `❌ Chưa có GEMINI_API_KEY hợp lệ trong .env.` và dừng ngay từ đầu thay vì chạy lỗi giữa chừng.

### 2. Danh mục Script và Lệnh chạy kiểm thử Trạm 1 (Tuần 2)

#### a) Kiểm thử tự động toàn diện (`pytest` - 138 tests)

Kiểm tra tính đúng đắn của phân loại bảng mã, chất lượng CER ≤ 0.05, độ nguyên vẹn của dấu gạch nối và khả năng phục hồi glyph "ư":

```powershell
python -m pytest tests/test_encoding_pipeline.py -v
```

#### b) Đo chỉ số định lượng (Precision, Recall, F1, CER, FCR)

Đo lường trên tập Benchmark độc lập 67 mẫu:

```powershell
python script/pilot_tuan2/encoding_evaluation.py script/ground_truth_manual_verified.csv
```

#### c) Chạy dây chuyền Ingestion trên toàn bộ Corpus PDF

Quét 9 file PDF, phân loại trang scan/mất dấu và xuất bảng CSV tổng kết:

```powershell
python script/pilot_tuan2/run_ingestion_check.py
```

#### d) Đo thời gian tách từ tiếng Việt của VnCoreNLP

Đo tốc độ tách từ trung bình (s/trang) qua nhiều lượt chạy:

```powershell
python script/pilot_tuan2/measure_wordseg_time.py --file "data/corpus/normal_lichsudang_C1&2_60tr.pdf" --runs 5
```

#### e) Các công cụ chẩn đoán và Soi văn bản

* **Chẩn đoán bảng mã non-ASCII:** `python script/pilot_tuan2/diagnose_encoding_chars.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf"`
* **Soi chi tiết từng trang:** `python script/pilot_tuan2/xem_noi_dung_trang.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 1`
* **Xuất toàn bộ văn bản sau chuẩn hóa:** `python script/pilot_tuan2/check_all.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf" "results/full_text_inspect/output.txt"`

### 3. Danh mục Script và Lệnh chạy kiểm thử Trạm 2 (Tuần 3)

Cài thêm thư viện cho Trạm 2 (nếu chưa có trong venv):

```powershell
pip install sentence-transformers faiss-cpu transformers
```

#### a) Demo truy hồi end-to-end

Chạy toàn bộ luồng: đọc PDF → chunking (page-aware + bridge chunk) → embedding → FAISS → trả lời top-k kèm số trang. Tự động lưu log ra `results/tuan3_pilot/`:

```powershell
python script/pilot_tuan3/demo_retrieval.py "data/corpus/normal_hienphap_33tr.pdf" "Vai trò của Mặt trận Tổ quốc Việt Nam là gì?"
```

#### b) Đo thực nghiệm toàn corpus (số chunk, thời gian, RAM)

Chạy trên 7 file PDF hợp lệ của corpus (loại file full-scan bị Trạm 1 từ chối và file `.docx` thuộc phần mở rộng), xuất CSV tổng hợp:

```powershell
python script/pilot_tuan3/measure_tuan3_corpus.py
```

**Lưu ý:** `py_vncorenlp` yêu cầu đường dẫn `save_dir` là đường dẫn **tuyệt đối**. Nếu chưa tải model VnCoreNLP hoặc gặp lỗi thiếu file `vi-vocab`, xem lại mục **1.5. Tải model VnCoreNLP** ở trên để tải/khắc phục trước khi chạy các script Trạm 2.

---

### 4. Danh mục Script và Lệnh chạy kiểm thử Trạm 3 & 4 (Tuần 4)

Tuần 4 bổ sung Hybrid Search (BM25), hiệu chỉnh ngưỡng cosine (threshold sweep) và tích hợp Gemini QA/Citation. **Cập nhật lại `requirements.txt` trước khi chạy** để tải các thư viện mới (`rank-bm25`, `google-generativeai`, `python-dotenv`):

```powershell
pip install -r requirements.txt
```

Đừng quên tạo file `.env` chứa `GEMINI_API_KEY` theo hướng dẫn ở mục **1.6** trước khi chạy các bước gọi Gemini API (mục c, d bên dưới).

#### a) Chuẩn hóa Dev Set (34 câu answerable + unanswerable)

Gộp bộ 10 câu cũ (Tuần 2) với các câu mới bổ sung, chuẩn hóa về chung 1 schema, xuất ra `data/eval_sets/dev_questions_normalized.json`:

```powershell
python script/pilot_tuan4/normalize_dev_questions.py
```

#### b) Đo Retrieval thô trên toàn Dev Set (Hit@3)

Chạy retrieval (k=3) cho cả 2 cấu hình chunking (page_aware, fixed_size) trên toàn bộ dev set, xuất `results/tuan4_pilot/dev_retrieval_raw.csv`:

```powershell
python script/pilot_tuan4/run_dev_retrieval.py
```

#### c) Quét ngưỡng Cosine (Threshold Sweep)

Quét dải tau từ 0.30 đến 0.80, tính false abstention rate, false acceptance rate và Abstention Precision/Recall/F1 cho từng cấu hình, xuất `results/tuan4_pilot/threshold_sweep.csv` và `threshold_recommendation.csv`:

```powershell
python script/pilot_tuan4/threshold_sweep.py
```

#### d) Chạy Gemini QA & Citation Generation theo 2 kịch bản ngưỡng

Sinh câu trả lời kèm trích dẫn số trang cho toàn bộ dev set. Script nhận tham số dòng lệnh để chạy tách biệt từng kịch bản ngưỡng, tránh lẫn kết quả:

```powershell
# Kịch bản A — ngưỡng riêng theo sweep (page_aware=0.5, fixed_size=0.45)
python script/pilot_tuan4/run_dev_qa.py --tau-page-aware 0.5 --tau-fixed-size 0.45

# Kịch bản B — ngưỡng chung thấp hơn (tau=0.38 cho cả 2 cấu hình)
python script/pilot_tuan4/run_dev_qa.py --tau 0.38
```

Kết quả xuất ra tương ứng `results/tuan4_pilot/dev_qa_results_scenario1_tau038.csv` và `dev_qa_results_scenario2_sweep.csv` (cột `answer_correctness_manual` cần chấm tay sau khi chạy xong).

**Lưu ý:** cả 2 lệnh ở mục d) đều gọi Gemini API thật (68 lượt/lệnh), cần `GEMINI_API_KEY` hợp lệ trong `.env` và có thể phát sinh chi phí/giới hạn quota (RPD) theo tier tài khoản Gemini đang dùng.

---

## Trạng thái tiến độ

* [x] **Tuần 1:** Chốt câu hỏi nghiên cứu, thiết lập Corpus 9 file, đo thực nghiệm giới hạn an toàn 60 trang / 20 MB.
* [x] **Tuần 2:** Hoàn thiện Trạm 1 (PDF Ingestion, Scan Detector, Text Normalizer đạt F1 = 1.000, CER = 0.003, FCR = 0.000 trên 67 mẫu benchmark độc lập và 166 trang corpus sạch; 138/138 pytest passed), đo thời gian tách từ (0.0154s/trang), hoàn thành 10 câu dev set.
* [x] **Tuần 3:** Hoàn thiện Trạm 2 (page-aware chunking với bridge chunk, fixed-size baseline, embedding `vietnamese-bi-encoder`, FAISS `IndexFlatIP`); chạy thành công trên 7/7 file corpus hợp lệ (1313 chunk page-aware gồm 263 bridge chunk, 1155 chunk fixed-size baseline); demo truy hồi end-to-end top-3 đúng chủ đề và đúng số trang.
* [x] **Tuần 4:** Hoàn thiện Trạm 3 (Hybrid Search BM25+Dense qua RRF) và Trạm 4 cơ bản (Gemini QA & Citation Generation); mở rộng dev set lên 34 câu (23 answerable / 11 unanswerable); quét ngưỡng cosine trên dải 0.30-0.80 và chạy song song 2 kịch bản ngưỡng (68 lượt Gemini/kịch bản); Hit@3 page_aware = 95.7% (+4.4pp so với fixed_size) → khóa page-aware làm kiến trúc mặc định.
* [ ] **Tuần 5:** Khóa ngưỡng tau chính thức, hoàn thiện Baseline Long-Context, khóa Test Set độc lập với Dev Set, kết nối luồng end-to-end (upload PDF → hỏi đáp → trích dẫn trang).
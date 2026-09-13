# Chat with Your PDF

Ứng dụng hỏi-đáp tài liệu PDF tiếng Việt có trích dẫn số trang, sử dụng kỹ thuật RAG (Retrieval-Augmented Generation).

- **Sinh viên thực hiện:** Võ Thành Phước — MSSV: 079205022977
- **Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Tiến
- **Thời gian thực hiện:** Tuần 1 – Tuần 8
- **Mã nguồn:** <https://github.com/KaitoP1912/chat-with-your-pdf>

Đề tài tập trung trả lời câu hỏi nghiên cứu: *chunking theo ranh giới trang có cải thiện độ chính xác trích dẫn số trang so với (a) chunking cố định không theo trang, và (b) không dùng truy hồi (đưa nguyên văn bản vào ngữ cảnh dài của Gemini) hay không?*

Phạm vi chính thức: PDF tiếng Việt có text layer. Hỗ trợ Word/.docx là phần mở rộng, không thuộc cam kết chính thức.

---

## 1. Kiến trúc hệ thống

| Trạm | Module | Mô tả |
|---|---|---|
| 1 — Ingestion | `source/ingestion/pdf_loader.py`, `scan_detector.py`, `text_normalizer.py` | Đọc PDF theo trang (`pdfplumber`), giữ metadata số trang gốc; phân loại `TEXT`/`LOW_TEXT`/`SCAN`/`EMPTY`/`MIXED_SCAN`/`FULL_SCAN`; chuẩn hóa bảng mã cũ (TCVN3/VNI → Unicode NFC) bằng Candidate Ranking + Lexicon Scoring; khôi phục lỗi glyph "ư" rớt; phát hiện trang mất dấu |
| 2 — Chunking & Indexing | `source/retrieval/chunker.py`, `word_segmenter.py`, `vectorstore.py` | Tách từ tiếng Việt (VnCoreNLP, chỉ annotator `wseg`); 2 chiến lược: `page_aware` (theo trang, bridge chunk 128 từ mỗi bên tại ranh giới trang, chunk tối đa 320 token, overlap 30 từ) và `fixed_size` (170 từ/chunk, không theo trang, baseline so sánh); embedding bằng `bkai-foundation-models/vietnamese-bi-encoder`; **Hybrid Search: FAISS (dense, `IndexFlatIP`) + BM25 (sparse), kết hợp qua Reciprocal Rank Fusion (RRF)** |
| 3 — Generation | `source/qa/qa_generator.py` | Gọi Gemini sinh câu trả lời; citation lấy từ metadata chunk, không cho LLM tự bịa số trang; abstention 2 tầng (ngưỡng retrieval `tau` + model tự nhận biết); gửi kèm ảnh trang cho các trang biểu đồ/bảng phức tạp đã xác định thủ công; cơ chế retry cho lỗi 429/503 |
| 4 — UI | `script/app_ui.py`, `script/app_ui_style.py` | Streamlit, 1 tài liệu/phiên, chat nhiều lượt, trích dẫn dạng chip, cảnh báo trang scan, cảnh báo trang mất dấu tiếng Việt |
| 5 — Evaluation | `script/pilot_tuan*/` | Hit@k, citation accuracy, false acceptance/refusal, answer correctness, latency, token, bridge-case riêng |

### Tham số đã khóa (`config.py`, ổn định từ 28/8/2026)

```
MODEL_NAME               = gemini-3.5-flash-lite
TAU                       = 0.38
TOP_K_GENERATION          = 15
CHUNK_MAX_TOKENS          = 320   (page_aware)
FIXED_CHUNK_WORDS         = 170   (fixed_size baseline)
CHUNK_OVERLAP_WORDS       = 30
BRIDGE_WORDS_EACH_SIDE    = 128
DEFAULT_CHUNKING_STRATEGY = page_aware
EMBED_MODEL_NAME          = bkai-foundation-models/vietnamese-bi-encoder
GENERATION_TEMPERATURE    = 0.0
```

Xem đầy đủ lịch sử khóa hệ thống (bằng chứng thời gian, thay đổi sau khóa) tại `report/tuan_7/BaoCao_ChatWithYourPDF_Tuan07_VoThanhPhuoc.md`, mục 2.

---

## 2. Cấu trúc thư mục

```
chat-with-your-pdf/
├── config.py                    # Tham số đã khóa
├── SYSTEM_LOCK.md                # Khóa cứng hệ thống (Tuần 6), xác minh Tuần 7
├── requirements.txt
├── .env                          # GEMINI_API_KEY (tự tạo, KHÔNG commit)
│
├── data/
│   ├── corpus/                   # 9 file PDF/docx gốc, khóa từ Tuần 1
│   ├── eval_sets/                 # dev_questions_normalized.json (34 câu)
│   │                              # test_questions.json (50 câu, chính thức)
│   └── held_out/                  # Dữ liệu thử nghiệm khắc phục lỗi, KHÔNG dùng đánh giá chính thức
│
├── source/
│   ├── ingestion/                 # Trạm 1
│   ├── retrieval/                  # Trạm 2-3 (chunking, embedding, hybrid search)
│   ├── qa/                        # Trạm 4 (Gemini QA, citation, abstention)
│   ├── evaluation/                 # Module đo Hit@k, CER, citation accuracy
│   └── ui/
│
├── script/
│   ├── app_cli.py                 # Demo CLI, 1 câu hỏi qua dòng lệnh
│   ├── app_ui.py                  # Ứng dụng Streamlit chính thức (ENTRY POINT chính)
│   ├── app_ui_style.py
│   ├── pilot_tuan1/ … pilot_tuan5/ # Script thực nghiệm/đánh giá từng tuần
│   ├── pilot_tuan6/                # run_test_qa.py, aggregate_results.py, chấm điểm
│   └── pilot_tuan7/
│       └── audit_scripts/          # Script kiểm chứng mở rộng Test Set 25→50 câu (có README riêng)
│
├── results/
│   └── tuan1_pilot/ … tuan6_pilot/  # CSV kết quả, bảng tổng hợp từng tuần
│
├── report/
│   └── tuan_1/ … tuan_8/            # Báo cáo tiến độ theo tuần
│
├── tests/
│   └── test_encoding_pipeline.py    # 138 test hồi quy cho chuẩn hóa bảng mã
│
├── chay_ung_dung.bat                 # Double-click để chạy giao diện, không cần gõ lệnh
└── vncorenlp_models/                 # Model VnCoreNLP (tải riêng, xem mục 3.3)
```

---

## 3. Cài đặt

### 3.1. Môi trường ảo

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn kích hoạt:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 3.2. Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Model VnCoreNLP (bắt buộc, tải riêng — không có trong repo)

> **Quan trọng:** `save_dir` bắt buộc phải là đường dẫn **tuyệt đối**. Đường dẫn tương đối gây lỗi JVM khó hiểu (`java.lang.NoClassDefFoundError: vn/pipeline/VnCoreNLP`) — đã xác nhận thực tế nhiều lần trong quá trình phát triển.

```powershell
python -c "import py_vncorenlp, os; py_vncorenlp.download_model(save_dir=os.path.abspath('vncorenlp_models'))"
```

Nếu thiếu file `vi-vocab` sau khi tải (lỗi tải model đôi khi gặp trên Windows):

```powershell
New-Item -ItemType Directory -Path "vncorenlp_models\models\wordsegmenter" -Force
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab" -OutFile "vncorenlp_models\models\wordsegmenter\vi-vocab"
```

Kiểm tra đã cài đúng:

```powershell
python -c "import py_vncorenlp, os; s = py_vncorenlp.VnCoreNLP(save_dir=os.path.abspath('vncorenlp_models'), annotators=['wseg']); print(s.word_segment('Việt Nam là một quốc gia Đông Nam Á'))"
```

### 3.4. Gemini API key

Tạo file `.env` ở thư mục gốc:

```
GEMINI_API_KEY=dán_API_key_của_bạn
```

Lấy key miễn phí tại [Google AI Studio](https://aistudio.google.com/app/apikey). Không commit `.env` lên Git.

---

## 4. Chạy ứng dụng — Hướng dẫn cho người dùng cuối (không cần biết code)

### Cách 1 — Double-click (đơn giản nhất, dùng khi demo/chấm điểm)

Double-click file **`chay_ung_dung.bat`** ở thư mục gốc. Cửa sổ terminal tự mở, tự kích hoạt môi trường, tự chạy Streamlit, trình duyệt tự mở giao diện.

### Cách 2 — Dòng lệnh (khi cần thấy log/debug)

```powershell
python -m streamlit run script/app_ui.py --server.fileWatcherType none
```

Giới hạn khi sử dụng: PDF có text layer, ≤ 20 MB, ≤ 60 trang, 1 tài liệu/phiên làm việc. Giao diện tự động cảnh báo nếu phát hiện trang scan hoặc trang mất dấu tiếng Việt trong file vừa upload.

### Demo CLI nhanh (không cần mở giao diện)

```powershell
python script/app_cli.py --pdf data/corpus/normal_hienphap_33tr.pdf --question "Nhiệm kỳ Quốc hội là bao nhiêu năm?" --vncorenlp_dir "$(Resolve-Path vncorenlp_models)"
```

---

## 5. Tái tạo đánh giá chính thức (Test Set 50 câu)

> Tất cả lệnh dưới đây **bắt buộc** truyền `--vncorenlp_dir` đường dẫn tuyệt đối — thiếu tham số này sẽ gặp lại lỗi JVM ở mục 3.3.

```powershell
$vncorenlp = (Resolve-Path vncorenlp_models).Path
```

**Chạy `page_aware`:**

```powershell
python script/pilot_tuan6/run_test_qa.py --strategy page_aware --test-set data/eval_sets/test_questions.json --vncorenlp_dir $vncorenlp --limit 0 --sleep 5.0 --out results/tuan6_pilot/test_qa_results_page_aware.csv
```

**Chạy `fixed_size`:**

```powershell
python script/pilot_tuan6/run_test_qa.py --strategy fixed_size --test-set data/eval_sets/test_questions.json --vncorenlp_dir $vncorenlp --limit 0 --sleep 5.0 --out results/tuan6_pilot/test_qa_results_fixed_size.csv
```

**Chạy `longcontext`** (không qua retrieval, không cần `--vncorenlp_dir`):

```powershell
python script/pilot_tuan5/run_longcontext_baseline.py --dev-set data/eval_sets/test_questions.json --limit 0 --sleep 5.0 --out results/tuan6_pilot/test_qa_results_longcontext.csv
```

*Tham số `--sleep 5.0` giảm nguy cơ lỗi rate-limit 429 (model flash-lite giới hạn 15 request/phút). Nếu 1 lệnh bị ngắt giữa chừng, chạy lại đúng lệnh đó kèm `--resume` để tiếp tục từ chỗ dở dang, không cần chạy lại từ đầu.*

**Chấm tay `answer_correctness`** (bắt buộc trước khi tổng hợp — dùng 3 nhãn `full`/`partial`/`wrong`):

```powershell
python script/pilot_tuan6/create_grading_template.py
python script/pilot_tuan6/semi_auto_grade.py
# Mở results/tuan6_pilot/grading_filled_auto.csv, đọc lại TỪNG dòng bằng mắt trước khi dùng
python script/pilot_tuan6/apply_grading_to_results.py --template results/tuan6_pilot/grading_filled_auto.csv
```

**Tổng hợp bảng so sánh 3 cấu hình:**

```powershell
python script/pilot_tuan6/aggregate_results.py
```

Kết quả ghi vào `results/tuan6_pilot/bang_so_sanh_3_cau_hinh.csv`.

---

## 6. Kết quả đánh giá chính thức (Test Set 50 câu: 34 answerable gồm 4 bridge case, 16 unanswerable)

| Chỉ số | page_aware (đề xuất) | fixed_size (baseline) | longcontext (baseline) |
|---|---|---|---|
| Hit@3 | **94,1%** | 88,2% | n/a |
| Citation accuracy | **100,0%** | 90,0% | 79,4% |
| False acceptance rate | 6,2% | 6,2% | **0,0%** |
| False refusal rate | 2,9% | 11,8% | **0,0%** |
| Answer correctness (full/partial/wrong) | 82,4% / 14,7% / 2,9% | 74,2% / 25,8% / 0,0% | 88,2% / 11,8% / 0,0% |
| Latency p50 | 1,78s | **1,40s** | 1,29s |
| Token trung bình | 4.197 | **2.711** | 41.417 |
| Bridge-case Hit@3 | **100% (4/4)** | 100% (4/4) | n/a |

Page-aware cải thiện 5,9 điểm % Hit@3 và 10 điểm % citation accuracy so với fixed-size; giữ citation accuracy cao hơn 20,6 điểm % so với long-context trong khi chỉ tốn ~10% lượng token. Chi tiết đầy đủ và error analysis: `report/tuan_7/BaoCao_ChatWithYourPDF_Tuan07_VoThanhPhuoc.md`.

---

## 7. Trạng thái tiến độ

- [x] Tuần 1 — Câu hỏi nghiên cứu, corpus 9 file, giới hạn vận hành (thực nghiệm)
- [x] Tuần 2 — Trạm 1: ingestion, scan detection, chuẩn hóa bảng mã (F1=1.000)
- [x] Tuần 3 — Trạm 2: page-aware chunking, bridge chunk, FAISS
- [x] Tuần 4 — Trạm 3-4, oracle-context test, khóa k=15+chunk=320
- [x] Tuần 5 — Hybrid Search, long-context baseline, khóa toàn bộ tham số
- [x] Tuần 6 — Giao diện Streamlit, khóa cứng hệ thống, kiểm thử 9 file corpus
- [x] Tuần 7 — Đánh giá chính thức Test Set 50 câu, error analysis, xác minh lịch sử khóa
- [x] Tuần 8 — Báo cáo tổng kết, README, cảnh báo mất dấu trên giao diện

---

## 8. Giới hạn kỹ thuật đã biết

| Giới hạn | Trạng thái |
|---|---|
| File VNI — một số chuyển đổi còn lỗi có quy luật | Đã xác nhận, ghi nhận, không sửa trong bản khóa hiện tại |
| File mixed-scan | Đã đóng — xác nhận hoạt động đúng thiết kế |
| Lỗi rớt dấu thanh trong text layer PDF (file Lịch sử Đảng) | Đã xác nhận nguyên nhân (đối chứng độc lập 2 công cụ đọc PDF); đã thử OCR nhưng chưa triệt để |
| File mất dấu tiếng Việt hoàn toàn | Gây fail retrieval; giao diện đã cảnh báo, chưa có fallback tự động sửa |
| PDF lớn | Giới hạn an toàn 60 trang/file (dựa trên stress-test thực nghiệm Tuần 1) |
| Bảng/biểu đồ phức tạp trong PDF | Có thể gây lỗi đọc nhầm số liệu liền kề; đã có cơ chế gửi ảnh trang hỗ trợ một phần |
| Word/.docx | Ngoài phạm vi cam kết chính thức, chưa triển khai |

Chi tiết đầy đủ: `report/tuan_7/` và `report/tuan_8/`.
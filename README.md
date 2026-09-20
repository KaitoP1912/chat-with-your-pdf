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
| 3 — Generation | `source/qa/qa_generator.py` | Gọi Gemini sinh câu trả lời; prompt yêu cầu Gemini trả JSON `{answer, used_sources}`; giao diện chỉ hiển thị nguồn thực sự được dùng, gộp theo trang; citation vẫn lấy từ metadata chunk, không cho LLM tự bịa số trang; abstention 2 tầng (ngưỡng retrieval `tau` + model tự nhận biết); gửi kèm ảnh trang cho các trang biểu đồ/bảng phức tạp đã xác định thủ công; cơ chế retry cho lỗi 429/503 |
| 4 — UI | `script/app_ui.py`, `script/app_ui_style.py` | Streamlit, 1 tài liệu/phiên, chat nhiều lượt, trích dẫn dạng chip, giao diện chỉ hiển thị nguồn thực sự được dùng và gộp theo trang, cảnh báo trang scan, cảnh báo trang mất dấu tiếng Việt |
| 5 — Evaluation | `script/pilot_tuan6/run_test_qa.py`, `aggregate_results.py` | Logic đo Hit@k, citation accuracy, false acceptance/refusal, answer correctness, latency, token, bridge-case riêng; `source/evaluation/` hiện chỉ là package placeholder |

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
│   └── evaluation/                 # Package placeholder, chưa chứa logic đánh giá chính
│
├── script/
│   ├── app_cli.py                 # Demo CLI, 1 câu hỏi qua dòng lệnh
│   ├── app_ui.py                  # Ứng dụng Streamlit chính thức (ENTRY POINT chính)
│   ├── app_ui_style.py
│   ├── debug_retrieval.py         # Debug retrieval và nội dung nguồn/chunk
│   ├── pilot_tuan1/ … pilot_tuan8/ # Script thực nghiệm, chấm điểm và tổng hợp từng tuần
│   ├── pilot_tuan6/                # run_test_qa.py, aggregate_results.py, chấm điểm
│   ├── pilot_tuan7/
│   │   └── audit_scripts/          # Script kiểm chứng mở rộng Test Set 25→50 câu (có README riêng)
│   └── pilot_tuan8/
│       ├── create_grading_template_tuan8.py
│       ├── semi_auto_grade_tuan8.py
│       └── aggregate_results_tuan8.py
│
├── results/
│   └── tuan1_pilot/ … tuan8/      # CSV kết quả, bảng tổng hợp từng tuần, bảng chính thức Tuần 8
│
├── report/
│   └── tuan_1/ … tuan_8/            # Báo cáo tiến độ theo tuần
│
├── tests/
│   ├── test_encoding_pipeline.py    # 138 test hồi quy cho chuẩn hóa bảng mã
│   └── test_pipeline_units.py       # Test đơn vị cho pipeline đánh giá/aggregation
│
├── chay_ung_dung.bat                 # Double-click để chạy giao diện, không cần gõ lệnh
└── vncorenlp_models/                 # Model VnCoreNLP (tải riêng, xem mục 3.3)
```

---

## 3. Cài đặt

> **Lưu ý cho Windows:** PowerShell mặc định dùng bảng mã cp1252, có thể
> gây lỗi `UnicodeEncodeError` khi chạy các script in tiếng Việt. Nếu gặp
> lỗi này, chạy lệnh sau 1 lần đầu mỗi phiên PowerShell trước khi chạy
> script Python:
> ```powershell
> $env:PYTHONIOENCODING = "utf-8"
> ```

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

VnCoreNLP chạy trên JVM, vì vậy cần cài sẵn **Java Runtime Environment (JRE) 1.8 trở lên** trước khi tải model. Kiểm tra bằng `java -version`; nếu chưa có Java, tải tại [java.com/download](https://www.java.com/download/).

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

Giới hạn khi sử dụng: PDF có text layer, ≤ 20 MB, ≤ 60 trang, 1 tài liệu/phiên làm việc. Giới hạn 60 trang được enforce cứng trong `source/ingestion/pdf_loader.py` (`MAX_PAGES = 60` và kiểm tra tổng số trang trước khi đọc). Mốc 90 giây là ngưỡng quan sát từ stress-test thực nghiệm Tuần 1, dùng để tính ra giới hạn 60 trang an toàn; đây không phải timeout chủ động ngắt tiến trình. Giao diện tự động cảnh báo nếu phát hiện trang scan hoặc trang mất dấu tiếng Việt trong file vừa upload.

### Demo CLI nhanh (không cần mở giao diện)

```powershell
python script/app_cli.py --pdf data/corpus/normal_hienphap_33tr.pdf --question "Nhiệm kỳ Quốc hội là bao nhiêu năm?" --vncorenlp_dir "$(Resolve-Path vncorenlp_models)"
```

---

## 5. Tái tạo đánh giá (Test Set 50 câu)

> Dưới đây là lần chạy chính thức đầu (12/09/2026) trên Test Set 50 câu.
>
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

### 5b. Chạy lại Tuần 8 (sau khi sửa hiển thị citation)

Đọc đúng `argparse` thực tế của `script/pilot_tuan6/run_test_qa.py`, `script/pilot_tuan5/run_longcontext_baseline.py` và 3 script trong `script/pilot_tuan8/` để chạy lại trên cùng Test Set 50 câu và ghi vào `results/tuan8/` với hậu tố `_50cau`. Các lệnh dưới đây không đoán tham số và không gọi API.

```powershell
$vncorenlp = (Resolve-Path vncorenlp_models).Path
python script/pilot_tuan6/run_test_qa.py --strategy page_aware --test-set data/eval_sets/test_questions.json --vncorenlp_dir $vncorenlp --limit 0 --sleep 5.0 --out results/tuan8/test_qa_results_page_aware_50cau.csv
python script/pilot_tuan6/run_test_qa.py --strategy fixed_size --test-set data/eval_sets/test_questions.json --vncorenlp_dir $vncorenlp --limit 0 --sleep 5.0 --out results/tuan8/test_qa_results_fixed_size_50cau.csv
python script/pilot_tuan5/run_longcontext_baseline.py --dev-set data/eval_sets/test_questions.json --limit 0 --sleep 5.0 --out results/tuan8/test_qa_results_longcontext_50cau.csv
python script/pilot_tuan8/create_grading_template_tuan8.py
python script/pilot_tuan8/semi_auto_grade_tuan8.py
python script/pilot_tuan8/aggregate_results_tuan8.py
```

`aggregate_results_tuan8.py` chỉ đọc CSV đã có và tính lại thống kê; nó không gọi API.

---

## 6. Kết quả đánh giá chính thức (Test Set 50 câu: 34 answerable gồm 4 bridge case, 16 unanswerable)

| Chỉ số | page_aware (đề xuất) | fixed_size (baseline) | longcontext (baseline) |
|---|---|---|---|
| Hit@3 | **94,1%** | 88,2% | n/a |
| Citation accuracy | **96,9%** | 87,1% | 76,5% |
| False acceptance rate | 6,2% | 6,2% | **0,0%** |
| False refusal rate | 5,9% | 8,8% | **0,0%** |
| Answer correctness (full/partial/wrong) | 85,3% / 8,8% / 0,0% | 82,4% / 5,9% / 2,9% | 97,1% / 2,9% / 0,0% |
| Latency p50 | 1,50s | **1,40s** | 1,42s |
| Token trung bình | 4.546 | **3.012** | 41.419 |
| Bridge-case Hit@3 | **100% (4/4)** | 100% (4/4) | n/a |

### Bảng đối chiếu lần chạy đầu → lần chạy lại

| Chỉ số | page_aware | fixed_size | longcontext |
|---|---|---|---|
| Hit@3 | 94,1% → 94,1% | 88,2% → 88,2% | n/a |
| Citation accuracy | 100,0% → 96,9% | 90,0% → 87,1% | 79,4% → 76,5% |
| False refusal | 2,9% → 5,9% | 11,8% → 8,8% | 0,0% → 0,0% |
| Token TB | 4.197 → 4.546 | 2.711 → 3.012 | 41.417 → 41.419 |

Ghi chú: bảng chính là lần chạy lại 20/9; Hit@3 và false acceptance không đổi. Citation giảm khoảng 3 điểm ở cả ba cấu hình sau khi rà soát lại logic hiển thị và từ chối. 

Page-aware cải thiện hơn fixed-size 5,9 điểm Hit@3 và 9,8 điểm citation; hơn long-context 20,4 điểm citation; khoảng 11% lượng token của long-context. Chi tiết đầy đủ và error analysis: `report/tuan_7/BaoCao_ChatWithYourPDF_Tuan07_VoThanhPhuoc.md`.

---

## 7. Trạng thái tiến độ

- [x] Tuần 1 — Câu hỏi nghiên cứu, corpus 9 file, giới hạn vận hành (thực nghiệm)
- [x] Tuần 2 — Trạm 1: ingestion, scan detection, chuẩn hóa bảng mã (F1=1.000)
- [x] Tuần 3 — Trạm 2: page-aware chunking, bridge chunk, FAISS
- [x] Tuần 4 — Trạm 3-4, oracle-context test, khóa k=15+chunk=320
- [x] Tuần 5 — Hybrid Search, long-context baseline, khóa toàn bộ tham số
- [x] Tuần 6 — Giao diện Streamlit, khóa cứng hệ thống, kiểm thử 9 file corpus
- [x] Tuần 7 — Đánh giá chính thức Test Set 50 câu, error analysis, xác minh lịch sử khóa
- [x] Tuần 8 — Chạy lại Test Set sau khi sửa hiển thị citation, báo cáo tổng kết, README, cảnh báo mất dấu trên giao diện

---

## 8. Giới hạn kỹ thuật đã biết

| Giới hạn | Trạng thái |
|---|---|
| File VNI — một số chuyển đổi còn lỗi có quy luật | Đã xác nhận, ghi nhận, không sửa trong bản khóa hiện tại |
| File mixed-scan | Đã đóng — xác nhận hoạt động đúng thiết kế |
| Lỗi rớt dấu thanh trong text layer PDF (file Lịch sử Đảng) | Đã xác nhận nguyên nhân (đối chứng độc lập 2 công cụ đọc PDF); đã thử OCR nhưng chưa triệt để |
| File mất dấu tiếng Việt hoàn toàn | Gây fail retrieval; giao diện đã cảnh báo, chưa có fallback tự động sửa |
| PDF lớn | Giới hạn cứng 60 trang/file, được enforce trong `source/ingestion/pdf_loader.py`; mốc 90 giây chỉ là ngưỡng quan sát từ stress-test thực nghiệm Tuần 1 để tính giới hạn an toàn, không phải timeout chủ động |
| Bảng/biểu đồ phức tạp trong PDF | Có thể gây lỗi đọc nhầm số liệu liền kề; đã có cơ chế gửi ảnh trang hỗ trợ một phần |
| Word/.docx | Ngoài phạm vi cam kết chính thức, chưa triển khai |

Chi tiết đầy đủ: `report/tuan_7/` và `report/tuan_8/`.
# BÁO CÁO TIẾN ĐỘ THỰC TẬP TỐT NGHIỆP — TUẦN 2

**Đề tài:** Xây dựng ứng dụng hỏi đáp tài liệu PDF tiếng Việt có trích dẫn số trang bằng kỹ thuật RAG (Chat with Your PDF).
**Sinh viên thực hiện:** Võ Thành Phước — MSSV: 079205022977.
**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Tiến.
**Thời gian báo cáo:** Tuần 2 (Hoàn thiện & Khóa Trạm 1).
**Quản lý mã nguồn (GitHub):** <https://github.com/KaitoP1912/chat-with-your-pdf>

---

## 1. Mục tiêu tuần 2

Mục tiêu cốt lõi của Tuần 2 là xây dựng hoàn chỉnh **Trạm 1 (PDF Ingestion & Text Normalization)**, đo thực nghiệm thời gian tách từ tiếng Việt, và chuẩn bị bộ câu hỏi phát triển (dev set) cho giai đoạn RAG:
1. Xây dựng luồng bóc tách PDF theo từng trang bằng `pdfplumber`, bảo toàn số trang gốc phục vụ trích dẫn số trang.
2. Phân loại cấu trúc tài liệu: phân biệt trang Text, trang Scan, trang Trống, trang Ít chữ và từ chối xử lý tài liệu Full Scan.
3. Nhận diện và chuẩn hóa bảng mã cũ (TCVN3, VNI-Windows) sang Unicode NFC thông qua cơ chế so khớp ứng viên (Candidate Ranking) kết hợp Lexicon Scoring và mật độ nguyên âm có dấu, loại bỏ hoàn toàn heuristic vá cứng.
4. Tự động phát hiện lỗi mất dấu tiếng Việt và khôi phục lỗi glyph font PDF rơi chữ "ư" (bao gồm dạng khoảng trắng và dạng dấu gạch nối).
5. Đo lường định lượng các chỉ số khoa học: Precision, Recall, F1, CER (Character Error Rate) và False Conversion Rate trên bộ Benchmark độc lập.
6. Đo thực nghiệm tốc độ tách từ của `py_vncorenlp` và hoàn thiện `dev_questions.json` (10 câu hỏi).

---

## 2. Công việc đã thực hiện

### 2.1. Module Ingestion & Tiền xử lý văn bản (`source/ingestion/`)

#### a) Trích xuất văn bản theo trang (`pdf_loader.py`)
- Trích xuất văn bản theo từng trang bằng `pdfplumber`, lưu kèm metadata `page_number` gốc.
- Áp dụng ngưỡng an toàn: dung lượng ≤ 20 MB và độ dài ≤ 60 trang (dựa trên bằng chứng thực nghiệm nghẽn CPU ở Tuần 1).
- Cắt biên trên/dưới nhẹ để loại bỏ header/footer rác và số trang in lặp lại.

#### b) Phân loại cấu trúc trang và tài liệu (`scan_detector.py`)
- Phân loại cấp trang: `TEXT`, `LOW_TEXT`, `SCAN`, `EMPTY`.
- Phân loại cấp tài liệu: `TEXT`, `MIXED_SCAN`, `FULL_SCAN`.
- Thực thi chính sách an toàn: Từ chối xử lý ngay lập tức các tài liệu `FULL_SCAN` (tránh lãng phí tài nguyên OCR/Embedding), giữ lại các trang khả dụng (`usable_pages`) của tài liệu `MIXED_SCAN`.

#### c) Chuẩn hóa bảng mã và khôi phục ký tự (`text_normalizer.py`)
- **Tách biệt bộ chuyển đổi:** Xây dựng module ánh xạ độc lập `convert_tcvn3_to_unicode()` (chuẩn TCVN 5712:1993) và `convert_vni_to_unicode()` (hỗ trợ đầy đủ tổ hợp 3 ký tự và chữ hoa).
- **Cơ chế Candidate Ranking:** Với mỗi trang, sinh đồng thời 3 bản ứng viên: `[original, as_tcvn3, as_vni]`.
- **Lexicon & Accent Scoring:** Chấm điểm dựa trên công thức kết hợp:

  $$\text{Score} = 0.8 \times \text{DictMatchRatio} + 0.2 \times \text{AccentCoverageRatio}$$

  sử dụng bộ từ điển tiếng Việt mở rộng `vietnamese_wordlist_external.txt` (10.622 từ), loại bỏ hoàn toàn việc hardcode từ vựng kiểm thử (No Data Leakage).
- **Quy tắc chuyển đổi có kiểm soát:** Chỉ chấp nhận chuyển mã khi Score ≥ 0.50 và (Score_candidate − Score_original) ≥ 0.12. Trường hợp mơ hồ (Δscore < 0.05), hệ thống tự động gán nhãn `unknown` và áp dụng *Zero-touch policy* (giữ nguyên 100% văn bản gốc).
- **Khôi phục glyph "ư" rớt:** Khôi phục độc lập dựa trên quy luật âm vị học tổng quát:

  `[Phụ âm đầu hợp lệ] + [khoảng trắng hoặc dấu '-'] + [nguyên âm tiếng Việt]`

  Xóa bỏ toàn bộ heuristic vá chuỗi cứng tùy tiện và bảo vệ 100% các dấu gạch nối thật trong văn bản.

---

### 2.2. Danh mục, vai trò & Cách chạy các Script trong `script/pilot_tuan2/`

Các script trong thư mục `script/pilot_tuan2/` được tổ chức thành 3 nhóm phục vụ kiểm thử và chẩn đoán:

#### Nhóm 1: Pipeline kiểm thử & Đánh giá định lượng cốt lõi
1. **`run_ingestion_check.py` [CỐT LÕI]:**
   * *Vai trò:* Tự động quét toàn bộ các file PDF trong `data/corpus/`, chạy dây chuyền qua `pdf_loader` → `scan_detector` → `text_normalizer` và xuất bảng tổng hợp CSV.
   * *Cách chạy:* `python script/pilot_tuan2/run_ingestion_check.py`
2. **`encoding_evaluation.py` [CỐT LÕI]:**
   * *Vai trò:* Tính toán khoa học các chỉ số Precision, Recall, F1 trên từng lớp nhãn (`tcvn3`, `vni`, `original`, `unknown`), đo khoảng cách Levenshtein / CER và False Conversion Rate trên bộ dữ liệu kiểm chuẩn Ground Truth.
   * *Cách chạy:* `python script/pilot_tuan2/encoding_evaluation.py script/ground_truth_manual_verified.csv`
3. **`measure_wordseg_time.py` [CỐT LÕI]:**
   * *Vai trò:* Đo tốc độ tách từ của `py_vncorenlp` qua nhiều lượt lặp độc lập trên file 60 trang để tính thời gian trung bình (s/trang).
   * *Cách chạy:* `python script/pilot_tuan2/measure_wordseg_time.py --file "data/corpus/normal_lichsudang_C1&2_60tr.pdf" --runs 5`

#### Nhóm 2: Xuất toàn văn & Kiểm tra trực quan từng trang
4. **`check_all.py`:**
   * *Vai trò:* Trích xuất và chuẩn hóa toàn bộ các trang của 1 file PDF, lưu thành file văn bản thuần `.txt` trong `results/full_text_inspect/` để đọc soát bằng mắt thường.
   * *Cách chạy:* `python script/pilot_tuan2/check_all.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf" "results/full_text_inspect/tcvn3.txt"`
5. **`xem_noi_dung_trang.py`:**
   * *Vai trò:* Soi chi tiết sự biến đổi của 1 trang cụ thể qua từng trạm: Raw Text → Phân loại Scan → Normalized Text.
   * *Cách chạy:* `python script/pilot_tuan2/xem_noi_dung_trang.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 1 2`

#### Nhóm 3: Công cụ chẩn đoán thực nghiệm (Bằng chứng cho Báo cáo)
6. **`diagnose_encoding_chars.py`:** Thống kê tần suất ký tự lạ non-ASCII trong file mã cũ phục vụ dựng bảng tra `TCVN3_MAP`/`VNI_MAP`.
   * *Cách chạy:* `python script/pilot_tuan2/diagnose_encoding_chars.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf"`
7. **`diagnose_char_spacing.py`:** Đo khoảng cách tọa độ (kerning) giữa các ký tự và thử nghiệm các ngưỡng `x_tolerance`.
   * *Cách chạy:* `python script/pilot_tuan2/diagnose_char_spacing.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf" 2`
8. **`diagnose_space_width.py`:** Đo độ rộng (`width = x1 - x0`) của dấu cách rỗng để chứng minh hiện tượng rớt glyph "ư".
   * *Cách chạy:* `python script/pilot_tuan2/diagnose_space_width.py "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf" 2`
9. **`diagnose_scan_pages.py`:** Bóc tách các XObjects và vector curves để giải thích vì sao trang scan không chứa ảnh raster `page.images`.
   * *Cách chạy:* `python script/pilot_tuan2/diagnose_scan_pages.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 1 38`

---

### 2.3. Đánh giá định lượng trên bộ Benchmark độc lập

Đã xây dựng tập dữ liệu kiểm chuẩn độc lập mở rộng gồm **67 mẫu** (`script/ground_truth_manual_verified.csv`) bao phủ 4 nhóm: TCVN3 thật, VNI thật (gồm tiêu đề chữ in hoa), Unicode chuẩn, Mất dấu, Lỗi rớt glyph "ư" (dấu cách và dấu gạch nối), và các trường hợp Unknown. Bộ từ vựng tính điểm được tách biệt độc lập khỏi tập kiểm thử.

Kết quả đo đạc thực tế qua `script/pilot_tuan2/encoding_evaluation.py`:

```text
=== Classification metrics (label-level)
Label,Precision,Recall,F1
original,1.000,1.000,1.000
tcvn3,1.000,1.000,1.000
unknown,1.000,1.000,1.000
vni,1.000,1.000,1.000

=== Character-level conversion quality (CER)
Avg CER (all rows): 0.003 (0.3%)
Avg CER (only correctly classified rows): 0.003 (n=67)
False Conversion Rate: 0.000 (0/67)
```

* **Hệ thống Kiểm thử tự động (`pytest`):** Đã vượt qua **138/138 tests (100% PASSED)** trong `tests/test_encoding_pipeline.py`, bao gồm bài test cô lập lỗi dấu gạch nối TCVN3 và xác nhận quét qua toàn bộ **166 trang PDF sạch** trong Corpus thực tế mà không xảy ra hiện tượng chuyển đổi nhầm (False Conversion = 0.000).

---

### 2.4. Kiểm thử tổng hợp trên Corpus Tuần 2

Chạy script `script/pilot_tuan2/run_ingestion_check.py` trên toàn bộ 9 file PDF trong `data/corpus/`.

Kết quả ghi nhận tại `results/tuan2_pilot/ingestion_check_summary.csv`:

| File | Trạng thái | Tổng trang | Scan Status | Trang scan | Trang trống | Trang ít chữ | Bảng mã | Confidence | Warning | Mất dấu? | Thời gian |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `normal_hienphap_33tr.pdf` | OK | 33 | text | - | - | - | unicode_chuẩn | 0.86 | OK | False | 6.89s |
| `normal_lichsudang_C1&2_60tr.pdf` | OK | 60 | text | - | Trang 2 | - | unicode_chuẩn | 0.78 | OK | False | 22.18s |
| `normal_vinamilkbaocao2014_53tr.pdf` | OK | 53 | text | - | - | Trang 1 | unicode_chuẩn | 0.84 | OK | False | 23.25s |
| `nodiacritic_118-2025-qh15_20tr.pdf` | OK | 20 | text | - | - | - | unicode_chuẩn | 0.60 | OK | **True** | 2.91s |
| `oldenc_tcvn3_36-2024-qh15_53tr.pdf` | OK | 53 | text | - | - | Trang 53 | **tcvn3** | 0.79 | OK | False | 9.50s |
| `Test2.pdf` (TCVN3 test mở rộng) | OK | 45 | text | - | - | - | **tcvn3** | 0.80 | OK | False | 5.11s |
| `oldenc_vni_118-2025-qh15_23tr.pdf` | OK | 23 | text | - | - | - | **vni** | 0.85 | OK | False | 7.55s |
| `mixedscan_qcvn06_38tr.pdf` | OK | 38 | mixed_scan | 1, 38 | 3, 7, 9 | - | unicode_chuẩn | 0.82 | OK | False | 5.06s |
| `scan_nd238_14tr.pdf` | **REJECTED** | 14 | full_scan | 1-14 | - | - | N/A | N/A | File scan 100% | N/A | 0.30s |

*Ghi chú: các cột "Trang scan/Trang trống/Trang ít chữ" liệt kê đúng số trang cụ thể, không phải số lượng.*
---

### 2.5. Đo thực nghiệm thời gian tách từ (`py_vncorenlp`)

Thực thi script `script/pilot_tuan2/measure_wordseg_time.py` trên file 60 trang `normal_lichsudang_C1&2_60tr.pdf` qua 5 lượt đo độc lập:

- Lượt 1: `0.0214 s/trang`
- Lượt 2: `0.0150 s/trang`
- Lượt 3: `0.0136 s/trang`
- Lượt 4: `0.0113 s/trang`
- Lượt 5: `0.0119 s/trang`
- **Trung bình 5 lượt:** ≈ **0.0154 s/trang** (Tổng thời gian xử lý 60 trang ≈ 0.92s, hoàn toàn không gây nghẽn pipeline).

Log chi tiết lưu tại: `results/tuan2_pilot/wordseg_time_output.txt`.

---

## 3. Tổng kết & Đối chiếu mục tiêu Tuần 2

| Hạng mục cam kết | Trạng thái | Kết quả / Minh chứng |
| --- | --- | --- |
| **Bóc tách PDF theo trang** | **ĐẠT** | `source/ingestion/pdf_loader.py` bảo toàn số trang gốc. |
| **Phân loại Scan / Empty / Text** | **ĐẠT** | `source/ingestion/scan_detector.py` nhận diện chính xác mixed scan và từ chối full scan. |
| **Chuẩn hóa TCVN3/VNI & Glyph "ư"** | **ĐẠT** | `source/ingestion/text_normalizer.py` đạt F1 = 1.000, CER = 0.003, FCR = 0.000. |
| **Phát hiện văn bản mất dấu** | **ĐẠT** | Gắn cờ chính xác `likely_missing_diacritics = True` trên file `nodiacritic`. |
| **Đo thời gian tách từ tiếng Việt** | **ĐẠT** | `results/tuan2_pilot/wordseg_time_output.txt` đạt trung bình 0.0154s/trang. |
| **Soạn bộ Dev Set** | **ĐẠT** | Hoàn thành 10 câu hỏi tại `data/eval_sets/dev_questions.json`. |
| **Hệ thống Kiểm thử tự động** | **ĐẠT** | Đạt **138/138 tests PASSED** trong `tests/test_encoding_pipeline.py`. |

---

## 4. Kế hoạch triển khai Tuần 3 (RAG Pipeline)

Trạm 1 đã hoàn tất và được khóa an toàn. Trọng tâm Tuần 3 sẽ chuyển sang xây dựng **Trạm 2 (Page-aware Chunking, Embedding & Indexing)**:

1. **Module `source/retrieval/chunker.py`:** Xây dựng bộ chia đoạn phân cấp (Fixed-size Chunking và Page-aware Chunking), bảo toàn metadata `doc_id` và `page_number` trong từng chunk.
2. **Module `source/retrieval/vectorstore.py`:** Tích hợp mô hình Embedding tiếng Việt (`BKAI/bge-m3` hoặc tương đương) và thiết lập chỉ mục vector `FAISS` in-memory.
3. **Thực nghiệm đo đạc Tuần 3:** Đo lường số lượng chunk, dung lượng bộ nhớ RAM và thời gian lập chỉ mục vector trên toàn bộ văn bản sạch của Corpus.
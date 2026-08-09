# Chốt kết quả Tuần 2 - Trạm 1 (Ingestion) và định hướng kỹ thuật Tuần 3

## 1. Kết quả thực nghiệm Tuần 2

### 1.1. PDF Loader
- `source/ingestion/pdf_loader.py` trích xuất text theo từng trang bằng `pdfplumber`.
- Pipeline giữ số trang gốc, có giới hạn an toàn `<= 20 MB` và `<= 60 trang`.
- Loader crop nhẹ mép trên/dưới để giảm header/footer rác.
- Loader hiện còn ghi nhận thêm một số chỉ số phụ trợ cho hậu kiểm scan detector như số object vector và cờ lỗi đọc ảnh.

### 1.2. Scan Detector
- `source/ingestion/scan_detector.py` phân loại ở mức trang và mức tài liệu.
- Trạng thái cấp trang: `TEXT`, `LOW_TEXT`, `SCAN`, `EMPTY`.
- Trạng thái cấp tài liệu: `TEXT`, `MIXED_SCAN`, `FULL_SCAN`.
- Kết quả chạy lại `run_ingestion_check.py` cho thấy:
  - `mixedscan_qcvn06_38tr.pdf` là `mixed_scan`, với trang scan `1, 38` và trang trống `3, 7, 9`.
  - `scan_nd238_14tr.pdf` bị từ chối đúng là `full_scan`.
  - `normal_lichsudang_C1&2_60tr.pdf` có trang trống ở trang `2`.

### 1.3. Text Normalizer
- `source/ingestion/text_normalizer.py` hiện tách tương đối rõ ba phần: loại page artifacts, thử candidate encoding TCVN3/VNI và chấm điểm, phát hiện văn bản mất dấu.
- Cơ chế nhận diện encoding là heuristic: module thử các candidate rồi chấm điểm bằng wordlist và một số tín hiệu phụ trợ.
- Có thêm heuristic `fix_missing_u_horn` để chèn lại chữ `ư` ở một số vị trí lỗi glyph/spacing, nhưng điều này không bảo đảm tạo thành từ tiếng Việt đúng chính tả trong mọi trường hợp.
- Không thể coi `encoding_confidence` là bằng chứng cho độ chính xác tuyệt đối.
- Ở trạng thái hiện tại, TCVN3/VNI đủ để chạy trên corpus Tuần 2, nhưng chưa có bằng chứng để gọi toàn bộ mapping là bảng mã chuẩn đầy đủ cho mọi nguồn dữ liệu.
- Một số mapping là heuristic/corpus-driven; chỉ nên mô tả là đủ dùng cho pipeline hiện tại, không nên mô tả là hoàn chỉnh.

### 1.4. Benchmark tách từ (`py_vncorenlp`)
- Script `script/pilot_tuan2/measure_wordseg_time.py` đã được chạy lại trên file 60 trang `normal_lichsudang_C1&2_60tr.pdf` với 5 lượt đo.
- Kết quả ghi nhận:
  - lượt 1: `0.0214 s/trang`
  - lượt 2: `0.0150 s/trang`
  - lượt 3: `0.0136 s/trang`
  - lượt 4: `0.0113 s/trang`
  - lượt 5: `0.0119 s/trang`
  - trung bình của 5 lượt: khoảng `0.0154 s/trang`
- Log đầy đủ đã được lưu vào `results/tuan2_pilot/wordseg_time_output.txt`.

### 1.5. Dev set
- `data/eval_sets/dev_questions.json` vẫn là bộ 10 câu hỏi phát triển dùng cho các tuần sau.
- Bộ này không làm thay đổi ingestion ở Tuần 2, chỉ phục vụ đánh giá truy hồi và trả lời ở tuần sau.

---

## 2. Kết quả chạy lại `run_ingestion_check.py`

Kết quả mới nhất được ghi trong `results/tuan2_pilot/ingestion_check_summary.csv`:

| File | Trạng thái | Tổng trang | Doc Scan Status | Trang scan | Trang trống | Trang ít chữ | Bảng mã | Confidence | Warning | Mất dấu? | Thời gian |
|---|---|---:|---|---|---|---|---|---:|---|---|---:|
| mixedscan_qcvn06_38tr.pdf | OK | 38 | mixed_scan | 1, 38 | 3, 7, 9 | - | unicode_chuẩn | 0.53 | OK | False | 5.31s |
| nodiacritic_118-2025-qh15_20tr.pdf | OK | 20 | text | - | - | - | unicode_chuẩn | 0.11 | OK | True | 3.24s |
| normal_hienphap_33tr.pdf | OK | 33 | text | - | - | - | unicode_chuẩn | 0.68 | OK | False | 7.61s |
| normal_lichsudang_C1&2_60tr.pdf | OK | 60 | text | - | 2 | - | unicode_chuẩn | 0.41 | OK | False | 24.04s |
| normal_vinamilkbaocao2014_53tr.pdf | OK | 53 | text | - | - | 1 | unicode_chuẩn | 0.51 | OK | False | 24.13s |
| oldenc_tcvn3_36-2024-qh15_53tr.pdf | OK | 53 | text | - | - | 53 | tcvn3 | 0.49 | OK | False | 8.29s |
| oldenc_vni_118-2025-qh15_23tr.pdf | OK | 23 | text | - | - | - | vni | 0.53 | OK | False | 4.54s |
| scan_nd238_14tr.pdf | REJECTED | 14 | full_scan | 1-14 | - | - | N/A | N/A | File scan 100% | N/A | 0.13s |

### Nhận xét thực tế
- Scan detector tách mixed scan và full scan đúng theo dữ liệu hiện hành.
- File `nodiacritic_118-2025-qh15_20tr.pdf` vẫn được gắn cờ `likely_missing_diacritics = True`.
- TCVN3 và VNI đều được nhận diện đúng trên 2 file controlled test.
- Confidence chỉ là chỉ số tham khảo, không nên diễn giải như độ chính xác tuyệt đối.
- Thời gian xử lý tăng ở file dài và file có nội dung phức tạp, nhưng vẫn ở mức chấp nhận được cho Tuần 2.

---

## 3. Đánh giá hiện trạng `text_normalizer.py`

- Module chạy được và đủ dùng cho pipeline tuần 2.
- Cơ chế chọn encoding vẫn dựa trên heuristic, không phải một phép chứng minh bảng mã chuẩn hoàn chỉnh.
- Chưa có đủ bằng chứng để khẳng định toàn bộ mapping TCVN3/VNI là đầy đủ cho mọi font hay mọi nguồn dữ liệu.
- Một số mapping trong file có nguồn gốc điều chỉnh từ corpus/controlled test, nên chỉ nên xem là đủ dùng cho pipeline hiện tại.
- Kết luận đúng ở thời điểm này: module hoạt động ổn, nhưng chưa thể mô tả là “bảng mã chuẩn đầy đủ”.

---

## 4. Định hướng Tuần 3

- Đo thử tokenizer thật của PhoBERT / `vietnamese-bi-encoder` trước khi chốt tham số chunk.
- Xây dựng Page-aware Chunker và Fixed-size Chunker.
- Tích hợp embedding và FAISS cho chỉ mục in-memory.
- Kiểm tra số chunk, bộ nhớ và thời gian tạo chỉ mục trên toàn bộ text sạch của corpus tuần 2.
- Phần đánh giá chất lượng truy hồi để sang Tuần 4 cùng `dev_questions.json`.

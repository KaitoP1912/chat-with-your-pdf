# BÁO CÁO TIẾN ĐỘ THỰC TẬP TỐT NGHIỆP

**TUẦN 2**

**Đề tài:** Xây dựng ứng dụng hỏi đáp tài liệu PDF tiếng Việt có trích dẫn số trang bằng kỹ thuật RAG (Chat with Your PDF).

**Sinh viên thực hiện:** Võ Thành Phước - MSSV 079205022977.

**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Tiến.

**Thời gian báo cáo:** Tuần 2.

**Quản lý mã nguồn:** Toàn bộ mã nguồn và các tệp thực hiện trong quá trình phát triển được quản lý và cập nhật trên GitHub tại: <https://github.com/KaitoP1912/chat-with-your-pdf>

---

## 1. Mục tiêu tuần 2

Mục tiêu tuần 2 là xây dựng Trạm 1 (Ingestion / Tiền xử lý văn bản), đo thực nghiệm thời gian tách từ tiếng Việt, và chuẩn bị bộ câu hỏi phát triển cho các tuần sau.

Cụ thể:
- Xây dựng luồng đọc PDF theo từng trang.
- Phân loại trang sạch, trang scan, trang trống và tài liệu scan toàn phần.
- Chuẩn hóa văn bản từ TCVN3/VNI về Unicode NFC và phát hiện văn bản thiếu dấu.
- Chạy kiểm thử tổng hợp trên corpus tuần 2 và lưu kết quả ra CSV.
- Đo thời gian tách từ `py_vncorenlp` bằng nhiều lượt chạy thực tế.
- Soạn xong `dev_questions.json` để dùng cho giai đoạn đánh giá truy hồi ở các tuần sau.

---

## 2. Công việc đã thực hiện

### 2.1. Trạm 1 (Ingestion / Tiền xử lý văn bản)

#### a) `source/ingestion/pdf_loader.py`
- Trích xuất text theo từng trang bằng `pdfplumber`.
- Giữ số trang gốc của PDF.
- Giữ giới hạn an toàn `<= 20 MB` và `<= 60 trang`.
- Crop nhẹ mép trên/dưới để giảm header/footer rác.
- Ghi nhận thêm một số chỉ số phụ trợ cho hậu kiểm scan detector, như số object vector và cờ lỗi đọc ảnh.

#### b) `source/ingestion/scan_detector.py`
- Phân loại ở mức trang: `TEXT`, `LOW_TEXT`, `SCAN`, `EMPTY`.
- Phân loại ở mức tài liệu: `TEXT`, `MIXED_SCAN`, `FULL_SCAN`.
- Tài liệu `FULL_SCAN` sẽ bị từ chối xử lý.
- Hàm `usable_pages()` chỉ giữ lại trang có thể dùng cho trạm sau.

#### c) `source/ingestion/text_normalizer.py`
- Loại bỏ số trang dính đầu dòng và các page artifacts đơn giản.
- Thử candidate encoding TCVN3/VNI và chấm điểm bằng wordlist tiếng Việt.
- Chuẩn hóa sang Unicode NFC sau khi chọn candidate phù hợp hơn.
- Phát hiện văn bản có dấu tiếng Việt bị mất bằng tỷ lệ ký tự có dấu.
- Có thêm heuristic `fix_missing_u_horn` để chèn lại chữ `ư` trong một số vị trí lỗi glyph/spacing, nhưng không phải bộ sửa lỗi chính tả tiếng Việt toàn diện.
- Cơ chế này hiện vẫn là heuristic; confidence chỉ là tín hiệu tham khảo, không phải bảo đảm tuyệt đối.
- Các mapping TCVN3/VNI hiện đủ để chạy trên corpus Tuần 2, nhưng chưa có đủ bằng chứng để gọi là bảng mã chuẩn đầy đủ cho mọi font/nguồn dữ liệu.

---

### 2.2. Đo thực nghiệm thời gian tách từ `py_vncorenlp`

Script `script/pilot_tuan2/measure_wordseg_time.py` được chạy lại trên file 60 trang `normal_lichsudang_C1&2_60tr.pdf` với 5 lượt đo.

Kết quả ghi nhận:
- Lượt 1: `0.0214 s/trang`
- Lượt 2: `0.0150 s/trang`
- Lượt 3: `0.0136 s/trang`
- Lượt 4: `0.0113 s/trang`
- Lượt 5: `0.0119 s/trang`
- Trung bình 5 lượt: khoảng `0.0154 s/trang`

Log chi tiết được lưu tại `results/tuan2_pilot/wordseg_time_output.txt`.

Kết quả cho thấy bước tách từ chạy ổn định và không tạo nút thắt đáng kể trong Trạm 1.

---

### 2.3. Bộ câu hỏi phát triển `data/eval_sets/dev_questions.json`

Đã soạn xong bộ 10 câu hỏi phát triển để dùng cho đánh giá ở các tuần sau.

Bộ này dùng cho kiểm thử truy hồi và trả lời, không ảnh hưởng đến ingestion ở Tuần 2.

---

### 2.4. Kiểm thử tổng hợp Trạm 1 trên corpus tuần 2

Đã chạy lại `script/pilot_tuan2/run_ingestion_check.py` trên toàn bộ 8 file PDF trong `data/corpus/`.

Kết quả tổng hợp được ghi tại `results/tuan2_pilot/ingestion_check_summary.csv`:

| File | Trạng thái | Tổng trang | Doc Scan Status | Trang scan | Trang trống | Trang ít chữ | Bảng mã | Confidence | Warning | Mất dấu? | Thời gian |
|---|---|---:|---|---|---|---|---|---:|---|---|---:|
| mixedscan_qcvn06_38tr.pdf | OK | 38 | mixed_scan | 1, 38 | 3, 7, 9 | - | unicode_chuẩn | 0.53 | OK | False | 5.31s |
| nodiacritic_118-2025-qh15_20tr.pdf | OK | 20 | text | - | - | - | unicode_chuẩn | 0.11 | OK | True | 3.24s |
| normal_hienphap_33tr.pdf | OK | 33 | text | - | - | - | unicode_chuẩn | 0.68 | OK | False | 7.61s |
| normal_lichsudang_C1&2_60tr.pdf | OK | 60 | text | - | 2 | - | unicode_chuẩn | 0.41 | OK | False | 24.04s |
| normal_vinamilkbaocao2014_53tr.pdf | OK | 53 | text | - | - | 1 | unicode_chuẩn | 0.51 | OK | False | 24.13s |
| oldenc_tcvn3_36-2024-qh15_53tr.pdf | OK | 53 | text | - | - | 53 | tcvn3 | 0.49 | OK | False | 8.29s |
| oldenc_vni_118-2025-qh15_23tr.pdf | OK | 23 | text | - | - | - | vni | 0.53 | OK | False | 4.54s |
| scan_nd238_14tr.pdf | REJECTED | 14 | full_scan | 1-14 | - | - | N/A | N/A | File scan toàn bộ | N/A | 0.13s |

Nhận xét thực tế:
- Mixed scan và full scan được nhận diện đúng theo trạng thái hiện hành của pipeline.
- File `nodiacritic_118-2025-qh15_20tr.pdf` vẫn bị gắn cờ mất dấu.
- Hai file controlled test `oldenc_tcvn3_36-2024-qh15_53tr.pdf` và `oldenc_vni_118-2025-qh15_23tr.pdf` được nhận diện đúng bảng mã tương ứng.
- Confidence chỉ phản ánh mức phù hợp tương đối của candidate hiện tại, không nên diễn giải như độ chính xác tuyệt đối.

---

## 3. Hậu kiểm và vấn đề phát hiện được

Trong quá trình rà soát Tuần 2, phát hiện scan detector ban đầu có thể nhầm một số trang scan vector hóa thành trang trống nếu chỉ dựa vào ảnh raster.

Sau đó, pipeline đã được hậu kiểm lại bằng cách đối chiếu trực quan và chạy lại script chẩn đoán trang scan. Kết quả hiện tại của `run_ingestion_check.py` cho thấy:
- `mixedscan_qcvn06_38tr.pdf` được gắn đúng là `mixed_scan`.
- Trang scan thật của file này là `1, 38`.
- Các trang trống thật là `3, 7, 9`.
- `scan_nd238_14tr.pdf` bị từ chối đúng là `full_scan`.

Điểm cần lưu ý là lỗi phân loại ban đầu chủ yếu ảnh hưởng đến độ chính xác của nhãn và log chẩn đoán, còn luồng `usable_pages()` vẫn loại bỏ trang không dùng được trước khi sang trạm sau.

---

## 4. Tình trạng `text_normalizer.py`

Hiện trạng thực tế của `text_normalizer.py` là:
- Module hoạt động được trên corpus Tuần 2.
- Có cơ chế thử candidate encoding và scoring, nhưng vẫn là heuristic.
- TCVN3/VNI hiện chưa thể mô tả là bảng mã chuẩn đầy đủ cho mọi nguồn dữ liệu.
- Không nên đọc confidence là độ đúng tuyệt đối.
- Không có cơ sở để khẳng định mọi mapping đều là chuẩn; một phần là kết quả điều chỉnh theo dữ liệu kiểm thử hiện có.

Vì vậy, báo cáo chỉ nên kết luận ở mức: module chạy được, pipeline ổn định, nhưng vẫn còn limitation ở phần suy luận encoding.

---

## 5. Đối chiếu yêu cầu tuần 2

| Yêu cầu | Trạng thái | Minh chứng |
|---|---|---|
| Hoàn thiện Trạm 1 | Đạt | `source/ingestion/pdf_loader.py`, `source/ingestion/scan_detector.py`, `source/ingestion/text_normalizer.py` |
| Đo thời gian `py_vncorenlp` | Đạt | `results/tuan2_pilot/wordseg_time_output.txt` |
| Hoàn thiện dev set | Đạt | `data/eval_sets/dev_questions.json` |
| Kiểm thử toàn bộ corpus tuần 2 | Đạt | `results/tuan2_pilot/ingestion_check_summary.csv` |

---

## 6. Kết luận tuần 2

Tuần 2 đã xây dựng xong trạm ingestion và có kết quả kiểm thử lại trên toàn bộ corpus tuần 2. Hệ thống hiện có thể:
- đọc và tiền xử lý PDF theo trang,
- phân loại trang scan/trống/tài liệu scan toàn phần,
- chuẩn hóa một phần dữ liệu mã cũ về Unicode,
- phát hiện văn bản thiếu dấu,
- lưu kết quả kiểm thử ra CSV,
- đo thời gian tách từ thực tế.

Tuy nhiên, phần text normalization vẫn còn limitation ở chỗ chưa thể chứng minh toàn bộ mapping TCVN3/VNI là bảng mã chuẩn đầy đủ cho mọi nguồn dữ liệu. Vì vậy, báo cáo tuần 2 chỉ nên mô tả đây là cơ chế heuristic đủ dùng cho corpus hiện tại, chưa phải lời khẳng định tuyệt đối.

---

## 7. Định hướng tuần 3

- Đo thử tokenizer thật của PhoBERT / `vietnamese-bi-encoder` trước khi chốt tham số chunk.
- Xây dựng Page-aware Chunker và Fixed-size Chunker.
- Tích hợp embedding và FAISS cho chỉ mục in-memory.
- Kiểm tra số chunk, bộ nhớ và thời gian tạo chỉ mục trên toàn bộ text sạch của corpus tuần 2.
- Phần đánh giá chất lượng truy hồi sẽ để sang Tuần 4 cùng `dev_questions.json`.

# Chat With Your PDF

Ứng dụng hỏi-đáp tài liệu PDF tiếng Việt có trích dẫn số trang, dùng kỹ thuật RAG (Retrieval-Augmented Generation).
Đồ án thực tập tốt nghiệp - Võ Thành Phước (MSSV 079205022977).
Giảng viên hướng dẫn: ThS. Nguyễn Thanh Tiến.

## Kiến trúc tóm tắt

Pipeline hiện tại gồm các bước chính:
1. Trích xuất và tiền xử lý văn bản theo từng trang.
2. Phân loại trang sạch, trang scan, trang trống và tài liệu scan toàn phần.
3. Chuẩn hóa văn bản từ bảng mã cũ về Unicode NFC.
4. Chia đoạn theo ranh giới trang ở các tuần sau.
5. Mã hóa vector và tìm kiếm ngữ nghĩa.
6. Sinh câu trả lời kèm trích dẫn số trang.

Trạng thái hiện tại của dự án:
- Trạm 1 (Ingestion) đã chạy được và đã kiểm thử trên corpus tuần 2.
- Các phần chunking, embedding, FAISS và QA sẽ tiếp tục ở Tuần 3 trở đi.
- Bộ câu hỏi phát triển đã có sẵn trong `data/eval_sets/dev_questions.json`.

Câu hỏi nghiên cứu, corpus, rubric đánh giá và văn bản chốt kỹ thuật theo từng tuần: xem tại thư mục `report/`.

---

## Cấu trúc thư mục

```text
data/
  corpus/              # Corpus PDF tuần 1-2 để phát triển và kiểm thử
  eval_sets/           # Bộ câu hỏi dev/test
  session_index/       # Chỉ mục tạm theo phiên làm việc
  session_uploads/     # File người dùng tải lên trong phiên
source/
  ingestion/           # Trạm 1: PDF loader, Scan detector, Text normalizer
  retrieval/           # Embedding, FAISS, tìm kiếm ngữ nghĩa
  qa/                  # Gọi Gemini API sinh câu trả lời + trích dẫn
  evaluation/          # Đo Hit@k, độ chính xác câu trả lời, tỉ lệ trích dẫn đúng trang
  ui/                  # Giao diện người dùng
script/
  pilot_tuan1/         # Script đo thực nghiệm Tuần 1
  pilot_tuan2/         # Script kiểm thử Trạm 1 và đo thời gian tách từ Tuần 2
results/
  tuan1_pilot/         # Output các lần đo Tuần 1
  tuan2_pilot/         # Output báo cáo CSV Trạm 1 và log thời gian tách từ Tuần 2
report/                # Báo cáo tiến độ và văn bản chốt kỹ thuật theo tuần
vncorenlp_models/      # Bộ mô hình VnCoreNLP cục bộ
```

---

## Hướng dẫn chạy kiểm thử Tuần 2

### 1. Chuẩn bị môi trường

Kích hoạt môi trường ảo Python và cài đặt phụ thuộc:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo tệp `.env` tại thư mục gốc nếu cần dùng Gemini ở các tuần sau:

```env
GEMINI_API_KEY=your_key_here
```

> `py_vncorenlp` yêu cầu máy đã có Java JDK và cấu hình `JAVA_HOME`.

### 2. Khởi tạo mô hình VnCoreNLP

Nếu cần tải lại mô hình word segmenter:

```powershell
python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='./vncorenlp_models')"
```

### 3. Chạy kiểm thử Tuần 2

Kiểm thử Trạm 1 trên toàn bộ corpus:

```powershell
python script/pilot_tuan2/run_ingestion_check.py
```

Kết quả được ghi vào `results/tuan2_pilot/ingestion_check_summary.csv`.

Đo thời gian tách từ trên file 60 trang:

```powershell
python script/pilot_tuan2/measure_wordseg_time.py --file "D:\PHUOC\HK9\TTTN\chat-with-your-pdf\data\corpus\normal_lichsudang_C1&2_60tr.pdf" --save-dir "D:\PHUOC\HK9\TTTN\chat-with-your-pdf\vncorenlp_models" --runs 5
```

Log được nối vào `results/tuan2_pilot/wordseg_time_output.txt`.

---

## Trạng thái tiến độ dự án

- **Tuần 1:** Đã hoàn thành chốt câu hỏi nghiên cứu, corpus và rubric đánh giá.
- **Tuần 2:** Đã hoàn thành Trạm 1, đo thời gian tách từ, soạn dev set và chạy kiểm thử tổng hợp corpus.
- **Tuần 3:** Dự kiến làm chunking, embedding và lập chỉ mục FAISS.

---

## Ghi chú quan trọng

- `source/ingestion/text_normalizer.py` hiện có cơ chế thử candidate encoding và scoring, nhưng đây vẫn là heuristic.
- `fix_missing_u_horn` chỉ là heuristic sửa lỗi glyph/spacing cho một số trường hợp mất chữ `ư`, không phải bộ sửa lỗi chính tả tiếng Việt toàn diện.
- Confidence không nên đọc như độ chính xác tuyệt đối.
- TCVN3/VNI hiện đủ để chạy trên corpus Tuần 2, nhưng chưa có đủ bằng chứng để gọi là bảng mã chuẩn đầy đủ cho mọi nguồn dữ liệu.
- Các kết quả chi tiết của Tuần 2 nằm trong thư mục `results/tuan2_pilot/` và `report/`.

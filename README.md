# Chat With Your PDF

Ứng dụng hỏi-đáp tài liệu PDF/Word tiếng Việt có trích dẫn số trang, dùng kỹ thuật RAG (Retrieval-Augmented Generation).
Đồ án thực tập tốt nghiệp - Võ Thành Phước (MSSV 079205022977).
Giảng viên hướng dẫn: ThS. Nguyễn Thanh Tiến.

## Kiến trúc tóm tắt

Pipeline RAG một vòng: trích xuất & chuẩn hóa văn bản → chia đoạn theo ranh giới trang (chunking) → mã hóa vector (embedding) → tìm kiếm ngữ nghĩa (FAISS) → sinh câu trả lời kèm trích dẫn số trang (Gemini API).

So sánh 3 cấu hình trên cùng bộ câu hỏi test:
- **Đề xuất**: chunking theo ranh giới trang.
- **Baseline A**: chunking cố định, không theo trang.
- **Baseline B**: không dùng truy hồi — đưa nguyên văn bản vào ngữ cảnh dài (long-context) của Gemini.

Câu hỏi nghiên cứu, corpus và rubric đánh giá chi tiết: xem `report/chot_cau_hoi_nghien_cuu_tuan1.md`.

## Cấu trúc thư mục

```
data/
  corpus/            # 9 file mẫu (PDF/Word) dùng xuyên suốt để phát triển và đánh giá
  eval_sets/          # bộ câu hỏi dev/test (đang xây dựng)
  session_index/      # chỉ mục FAISS theo phiên làm việc (không lưu lâu dài)
  session_uploads/     # file người dùng tải lên trong phiên (không lưu lâu dài)
source/
  ingestion/          # loader PDF/Word, chuẩn hóa văn bản, chunking
  retrieval/           # embedding, FAISS, tìm kiếm ngữ nghĩa
  qa/                  # gọi Gemini API sinh câu trả lời + trích dẫn
  evaluation/          # đo Hit@k, độ chính xác câu trả lời, tỉ lệ trích dẫn đúng trang
  ui/                  # giao diện Streamlit
script/
  pilot_tuan1/         # 3 script đo thực nghiệm tuần 1 (xem bên dưới)
results/
  tuan1_pilot/         # output các lần đo tuần 1
report/                # báo cáo tiến độ theo tuần
```

## Cài đặt

```
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo file `.env` với nội dung:

```
GEMINI_API_KEY=your_key_here
```

## Chạy các script đo thực nghiệm tuần 1 (pilot)

```
python script/pilot_tuan1/measure_corpus_stats.py
python script/pilot_tuan1/measure_embedding_time.py
python script/pilot_tuan1/test_longcontext_baseline.py
```

Kết quả được lưu vào `results/tuan1_pilot/`. Script `measure_embedding_time.py` có thể chạy nhiều lần liên tiếp — mỗi lần chạy được nối thêm (append) vào cùng file kết quả, không ghi đè lần chạy trước, và tự in trung bình các lần đã có.

## Trạng thái hiện tại

- **Tuần 1**: đã chốt câu hỏi nghiên cứu, corpus mẫu (9 file), rubric đánh giá, và các giới hạn vận hành (125 trang / 90 giây, giới hạn token baseline B) dựa trên đo thực nghiệm. Chi tiết: `report/`.
- Các module trong `source/` mới khởi tạo, chưa cài đặt logic — dự kiến từ tuần 2 theo kế hoạch trong đề cương.
# Audit scripts

Các script trong thư mục này là bằng chứng cho quy trình kiểm chứng đã thực hiện qua 2 đợt: (A) sửa lỗi chấm điểm Tuần 7, (B) mở rộng Test Set 25→50 câu Tuần 7. Nội dung script được giữ nguyên khi di chuyển, không sửa lại.

## Đợt A — Sửa lỗi chấm điểm

| Script | Mục đích | Kết quả tạo ra |
|---|---|---|
| `check_mixedscan_questions.py` | Kiểm tra bộ câu hỏi có câu nào liên quan tài liệu mixed-scan/QCVN không. | In ra terminal; không ghi file. |
| `cross_check_page21.py` | Đối chiếu lỗi mất dấu trang 21 (Lịch sử Đảng) bằng PyMuPDF, độc lập với pdfplumber — xác nhận lỗi nằm trong PDF gốc, không phải lỗi code. | In toàn văn trang 21 và kết quả đối chiếu; không ghi file. |
| `fix_grading_manual.py` | Sửa tay 3 dòng chấm điểm sai (test_10, test_25 page_aware, test_25 fixed_size). | Ghi vào `grading_filled_auto.csv`. |
| `fix_grading_v2.py` | Sửa lại test_25 theo phản hồi GVHD: `wrong` → `partial`. | Ghi vào `grading_filled_auto.csv`. |
| `fix_grading_v3.py` | Sửa test_20 (fixed_size) thành `partial`. | Ghi vào `grading_filled_auto.csv`. |
| `analyze_eval_sets.py` | Phân tích cấu trúc/nội dung bộ câu hỏi đánh giá. | In ra terminal. |
| `check_eval_sets.py` | Kiểm tra tính hợp lệ, nhất quán của bộ câu hỏi. | In cảnh báo ra terminal. |

## Đợt B — Mở rộng Test Set 25→50 câu (kiểm chứng chống rò rỉ dữ liệu)

| Script | Mục đích | Kết quả tạo ra |
|---|---|---|
| `lap_vung_cam.py` | Trích toàn bộ trang/câu hỏi đã dùng trong dev set + 25 câu test cũ — lập "vùng cấm" để 25 câu mới không được trùng, tránh lặp lại lỗi rò rỉ dữ liệu đã phát hiện trước đó. | Ghi `vung_cam_trang_da_dung.json`. |
| `verify_cau_hoi_moi.py` | Kiểm tra 25 câu hỏi mới (Gemini soạn) có dùng đúng trang cấm không, kiểm tra bridge case có 2 trang liền kề thật không, kiểm tra trùng lặp nội bộ. | In danh sách lỗi ra terminal. |
| `tim_dung_trang.py` | Quét toàn bộ PDF tìm đúng trang chứa nội dung câu trả lời (xử lý cả PDF lỗi rớt dấu và layout nhiều cột) khi câu hỏi bị nghi ngờ sai trang. | Ghi `ket_qua_tim_trang.txt`. |
| `verify_tu_khoa.py` | Đối chiếu answer_reference với nội dung trang bằng % từ khóa trùng (không cần liền mạch) — khắc phục hạn chế của `tim_dung_trang.py` với PDF layout nhiều cột (Vinamilk). | Ghi `ket_qua_tu_khoa.txt`. |
| `gop_50_cau.py` | Gộp 25 câu mới (đã verify xong) vào `test_questions.json` cũ, đánh số `test_26`-`test_50`. | Ghi đè `data/eval_sets/test_questions.json` (50 câu). |
| `sua_dinh_dang_expected_page.py` | Chuẩn hóa `expected_page` về dạng list cho toàn bộ câu hỏi (25 câu mới ban đầu ghi số nguyên trần, gây lỗi khi chạy `run_test_qa.py`). | Ghi đè `test_questions.json`. |
| `ap_lai_4_diem_da_chot_v2.py` | Áp lại 4 nhãn chấm điểm đã chốt thủ công (test_10, test_25 x2, test_20) sau khi bị `semi_auto_grade.py` chạy lại đè mất khi mở rộng lên 50 câu. | Sửa trực tiếp 3 file `test_qa_results_*.csv`. |
| `check_4_diem_da_sua.py` | Kiểm tra nhanh 4 điểm đã chốt ở trên có đang đúng nhãn hay không, dùng để phát hiện việc bị đè mất. | In ra terminal. |

## Kết quả cuối cùng của cả 2 đợt

Test Set chính thức: `data/eval_sets/test_questions.json`, 50 câu (34 answerable gồm 4 bridge case, 16 unanswerable). Kết quả đánh giá: `results/tuan6_pilot/bang_so_sanh_3_cau_hinh.csv`, xử lý tiếp bởi `script/pilot_tuan6/apply_grading_to_results.py` và `aggregate_results.py`. Chi tiết đầy đủ quy trình và error analysis: `report/tuan_7/BaoCao_ChatWithYourPDF_Tuan07_VoThanhPhuoc.md`.
# Chốt câu hỏi nghiên cứu, tài liệu mẫu và rubric — Tuần 1

## 1. Câu hỏi nghiên cứu trung tâm

Chunking theo ranh giới trang có cải thiện độ chính xác trích dẫn số trang so với (a) chunking cố định không theo trang, và (b) không dùng truy hồi (đưa nguyên văn bản vào ngữ cảnh dài của Gemini) hay không? Cải thiện bao nhiêu phần trăm?

## 2. Giới hạn vận hành (chốt từ đo thực nghiệm — xem script/pilot_tuan1/)

- Giới hạn số trang: 60 trang (hạ từ 125 trang dựa trên kết quả stress-test file 114 trang bị nghẽn CPU).
- Giới hạn dung lượng file input: 20 MB (ước lượng, không phải đo thực nghiệm như số trang/timeout — dựa trên cỡ PDF tiếng Việt dạng text thuần điển hình ~50-100 KB/trang, 60 trang × mức này còn dư nhiều dư địa cho file có ảnh/scan; đáp ứng góp ý của giảng viên về bổ sung giới hạn dung lượng. Sẽ đối chiếu lại với dung lượng thực tế của 9 file corpus khi làm module loader ở tuần 2, có thể điều chỉnh con số này).
- Timeout xử lý: 90 giây.
- Trần token giới hạn cho Baseline Long-Context: 100.000 token (đáp ứng nhận xét của thầy hướng dẫn).
- Căn cứ thực nghiệm: Bằng chứng nghẽn CPU (stress-test) — khi chạy thử nghiệm file normal_lichsudang_114tr.pdf (114 trang, 462 chunk) trên CPU local, thời gian embedding tốn tới 143,18 giây (trung bình 1,256 giây/trang). Điều này cho thấy giả định 125 trang/90 giây bị phá vỡ khi gặp văn bản mật độ chữ dày. Chốt giới hạn MVP mới: hạ giới hạn số trang an toàn xuống 60 trang/file. Đo lại thực nghiệm trên data/corpus/tests/normal_hienphap.pdf (33 trang, 101 chunk) qua 5 lần chạy độc lập: 0,648 / 0,789 / 0,648 / 0,887 / 0,695 giây/trang. Trung bình 5 lần: **0,733 giây/trang**. Ước tính cho 60 trang: 60 × 0,733s ≈ 44,00 giây, nằm an toàn dưới ngưỡng timeout 90 giây, còn dư khoảng 46 giây cho các bước còn lại (PDF parsing, tách từ py_vncorenlp, lập chỉ mục FAISS). Chi tiết đầy đủ cả 5 lần chạy: results/tuan1_pilot/embedding_time_output.txt.
- Giới hạn token baseline B (long-context): xác nhận qua đo thật — 33 trang Hiến pháp ≈ 20.400 token input (count_tokens Gemini). Áp dụng cùng file cho cả 3 cấu hình khi so sánh, không đổi model/API key giữa các lần chạy test.

## 3. Tài liệu mẫu (corpus chính thức — chốt ngày 31/07/2026, không đổi sau khi khóa test)

Đặt tại data/corpus/, gồm 9 file:

| # | File | Trang | Loại | Ghi chú nguồn |
|---|---|---|---|---|
| 1 | normal_hienphap_33tr.pdf | 33 | Chuẩn — luật | Hiến pháp CHXHCN VN (sửa đổi 2025), copyright-free theo Điều 15 Luật SHTT |
| 2 | normal_vinamilkbaocao2014_53tr.pdf | 53 | Chuẩn — báo cáo doanh nghiệp (có bảng/biểu đồ/cột) | Vinamilk, website chính thức — chỉ dùng nội bộ cho đồ án, không redistribute |
| 3 | normal_lichsudang_C1&2_60tr.pdf | 60 | Chuẩn — giáo trình văn xuôi | Giáo trình Lịch sử Đảng CSVN |
| 4 | mixedscan_qcvn06_38tr.pdf | 38 | Lỗi — scan từng phần (trang 1 scan, còn lại text) | QCVN 06:2022, Bộ Xây dựng — copyright-free (văn bản quy phạm pháp luật) |
| 5 | scan_nd238_14tr.pdf | 14 | Lỗi — scan toàn bộ | Nghị định 238/2026/NĐ-CP — copyright-free |
| 6 | oldenc_vni_118-2025-qh15_23tr.pdf | 23 | Lỗi — bảng mã VNI | Luật 118/2025/QH15 — copyright-free |
| 7 | oldenc_tcvn3_36-2024-qh15_53tr.pdf | 53 | Lỗi — bảng mã TCVN3 | Luật 36/2024/QH15 — copyright-free |
| 8 | nodiacritic_118-2025-qh15_20tr.pdf | 20 | Lỗi — mất dấu | Luật 118/2025/QH15 — copyright-free |
| 9 | word_118-2025-qh15_28tr.docx | 28 (số trang Word gốc, có thể lệch khi convert PDF) | Chuẩn — .docx gốc | Luật 118/2025/QH15, file Word gốc chính thức — copyright-free |

Vai trò sử dụng:

- File 1, 2, 3: dùng soạn bộ 60 câu hỏi dev/test chính (10 dev + 50 test), chia đều ra 3 file để tránh câu hỏi thiên lệch về 1 dạng cấu trúc văn bản.
- File 4-9: dùng test sanity-check các module xử lý edge case (scan detector, mixed-scan theo trang, encoding detector, diacritic detector, loader Word) - 1-3 câu hỏi/file, không tính vào bộ 60 câu chính.
- Loại trừ khi soạn câu hỏi: không lấy đáp án nằm trong bảng phức tạp hoặc hình ảnh không có text layer (áp dụng cho file 2 — báo cáo Vinamilk có biểu đồ/bảng) - ngoài phạm vi 8 tuần theo đề cương.
- File 4 (mixedscan) là căn cứ để chốt: scan detector phải chạy theo từng trang, không chạy 1 lần cho toàn file, để không bỏ sót cảnh báo hoặc từ chối nhầm cả file chỉ vì 1 trang bìa bị scan.

## 4. Rubric chấm

- Hit@3 (retriever): đúng nếu trong top-3 đoạn truy hồi có ít nhất 1 đoạn chứa đúng trang đáp án.
- Độ chính xác câu trả lời (3 mức):
  - Đúng hoàn toàn: khớp đầy đủ ý đáp án mẫu và trích dẫn đúng trang chứa đáp án.
  - Đúng một phần: đúng nội dung nhưng thiếu ý, hoặc đúng nội dung nhưng trích dẫn trang sai/thiếu.
  - Sai: nội dung sai lệch bản chất, bịa thông tin không có trong tài liệu, hoặc trả lời "không có thông tin" trong khi tài liệu có.
- Tỉ lệ trích dẫn đúng trang: số câu trả lời có trích dẫn khớp đúng trang / tổng số câu có trích dẫn.
- Nhận biết "không có thông tin": đo riêng tỉ lệ hệ thống trả lời đúng là không tìm thấy khi câu hỏi thực sự nằm ngoài tài liệu.
- Đánh giá riêng bridge chunk: trong tập câu hỏi test, tách riêng nhóm câu có đáp án trải 2 trang liền kề (tối đa 10% bộ test theo đề cương mục 4) — đo Hit@3 và tỉ lệ trích dẫn đúng trang chỉ trên nhóm này, so sánh với nhóm câu hỏi có đáp án nằm trọn 1 trang, để xác nhận đoạn cầu nối (bridge chunk) có thực sự cải thiện hay không, thay vì chỉ gộp chung vào chỉ số tổng.

## 5. Điều chỉnh file đề cương

- Tên đề tài: "Xây dựng ứng dụng hỏi đáp tài liệu PDF tiếng Việt có trích dẫn số trang bằng kỹ thuật RAG".
- Phạm vi: PDF có text layer là phạm vi bắt buộc; Word (qua LibreOffice) là phần mở rộng, chỉ làm nếu còn thời gian (tuần 7 trở đi), không phải cam kết chính thức song song với PDF. Corpus tuần 1 đã áp dụng đúng tinh thần này (file Word chỉ dùng sanity-check, không tính vào 60 câu hỏi chính).
- Thứ tự công việc cần sửa lại: 10 câu dev phải soạn xong trước tuần 4 (khuyến nghị làm song song với chuẩn bị corpus, ngay từ tuần 1-2); 50 câu test có thể soạn nốt trước khi khóa test ở tuần 5, theo đúng tinh thần "khóa model, prompt, threshold và test set" của khung tiến độ chung đề ra cho tuần 5.
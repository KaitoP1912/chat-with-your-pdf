# BÁO CÁO TIẾN ĐỘ THỰC TẬP TỐT NGHIỆP — TUẦN 7

**Đề tài:** Xây dựng ứng dụng hỏi đáp tài liệu PDF tiếng Việt có trích dẫn số trang bằng kỹ thuật RAG (Chat with Your PDF).
**Sinh viên thực hiện:** Võ Thành Phước — MSSV: 079205022977.
**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Tiến.
**Thời gian báo cáo:** Tuần 7 (Đánh giá chính thức trên Test Set 50 câu — Error Analysis).
**Quản lý mã nguồn (GitHub):** <https://github.com/KaitoP1912/chat-with-your-pdf>

---

## 1. Mục tiêu Tuần 7

1. Chạy đánh giá chính thức trên Test Set đã khóa, trên đúng 3 cấu hình: `page_aware`, `fixed_size`, `longcontext`.
2. Hoàn thành chấm tay `answer_correctness` (full/partial/wrong) cho toàn bộ câu answerable.
3. Xác nhận và ghi lại minh bạch lịch sử khóa hệ thống — chứng minh Test Set không được dùng để điều chỉnh tham số.
4. Viết error analysis cho các trường hợp trả lời sai/đáng chú ý.

---

## 2. Xác nhận lịch sử khóa hệ thống

| # | Yêu cầu | Bằng chứng |
|---|---|---|
| 1 | Commit/version tại thời điểm lần đầu mở Test Set | `config.py` ổn định từ **28/8/2026, 17:37**, không đổi kể từ đó đến nay. Commit chính thức tại `d40fc4d` (30/8/2026) |
| 2 | Cấu hình chính tại thời điểm đó | `tau=0.38`, `k=15`, `chunk_size=320`, chiến lược chính: `page_aware`, model: `gemini-3.5-flash-lite` |
| 3 | Thời điểm khóa hệ thống | `SYSTEM_LOCK.md` tạo **6/9/2026, 07:59** |
| 4 | Thay đổi thực hiện sau khi xem kết quả Test Set | (a) `app_cli.py` sửa lúc 6/9, 08:19 — chỉ thêm early-exit cho file scan hoàn toàn, không chạm code path của 3 file Test Set dùng (không phải file scan); (b) mở rộng Test Set từ 25 lên 50 câu (mục 2.1 dưới đây) |
| 5 | Bằng chứng thay đổi không xuất phát từ kết quả Test Set | Xem chi tiết bên dưới |

### 2.1. Về việc mở rộng Test Set từ 25 lên 50 câu

Kế hoạch ban đầu (Tuần 1) đặt mục tiêu 50 câu test. Do phát hiện rủi ro rò rỉ dữ liệu khi thử mở rộng lần đầu (25 câu "mới" trùng gần như nguyên văn dev set), đề tài đã tạm giữ 25 câu độc lập cho đánh giá sơ bộ. Ở Tuần 7, 25 câu bổ sung được soạn lại theo quy trình kiểm chứng nghiêm ngặt:

1. Lập danh sách "vùng cấm" bằng code — toàn bộ trang/chủ đề đã dùng trong dev set (34 câu) và 25 câu test cũ, để đảm bảo 25 câu mới không trùng lặp nội dung đã dùng để tinh chỉnh tham số ở Tuần 4-5.
2. Soạn câu hỏi mới (hỗ trợ bởi Gemini Pro) chỉ trong phạm vi trang chưa dùng, trộn đều 3 loại: câu 1 trang, câu bridge (vắt 2 trang liền kề), câu out-of-scope.
3. Xác minh bằng 3 lớp kiểm tra độc lập: (a) script đối chiếu tự động không trùng vùng cấm, (b) script đối chiếu nội dung trực tiếp với PDF gốc (xử lý cả trường hợp PDF bị lỗi rớt dấu và layout nhiều cột gây xáo trộn thứ tự trích xuất), (c) đọc tay đối chiếu trực tiếp các trường hợp còn nghi vấn.
4. Toàn bộ tham số hệ thống (`tau`, `k`, chunk size, prompt, model) **giữ nguyên không đổi** trong suốt quá trình này — chỉ mở rộng dữ liệu đánh giá, không tinh chỉnh hệ thống dựa trên kết quả quan sát được.

**Kết quả đối chiếu:** Hit@3 và Citation accuracy của `page_aware` trên 50 câu **giữ nguyên y hệt** so với 25 câu ban đầu (94,1% / 100%) — củng cố độ tin cậy của kết luận, không phải hiệu ứng ngẫu nhiên của cỡ mẫu nhỏ.

---

## 3. Kết quả đánh giá chính thức — Test Set (50 câu: 34 answerable gồm 4 bridge case, 16 unanswerable)

| Chỉ số | page_aware (đề xuất) | fixed_size (baseline) | longcontext (baseline) |
|---|---|---|---|
| Hit@3 (retriever) | **94,1%** | 88,2% | n/a |
| Citation accuracy | **100,0%** | 90,0% | 79,4% |
| False acceptance rate | 6,2% | 6,2% | **0,0%** |
| False refusal rate | 2,9% | 11,8% | **0,0%** |
| Answer correctness (full/partial/wrong) | 82,4% / 14,7% / 2,9% | 74,2% / 25,8% / 0,0% | 88,2% / 11,8% / 0,0% |
| Latency trung bình | 2,31s | 1,60s | 1,38s |
| Latency p50 | 1,78s | 1,40s | 1,29s |
| Token trung bình | 4.197 | **2.711** | 41.417 |
| Bridge-case Hit@3 | **100% (4/4)** | 100% (4/4) | n/a |

Chấm tay đầy đủ toàn bộ câu answerable cho cả 3 cấu hình.

### Trả lời câu hỏi nghiên cứu trung tâm

Trên bộ Test Set 50 câu, page-aware chunking cải thiện **5,9 điểm phần trăm Hit@3** (94,1% so với 88,2%) và **10 điểm phần trăm citation accuracy** (100% so với 90,0%) so với fixed-size chunking. Kết quả này giữ nguyên xu hướng đã quan sát ở 25 câu đầu, với cỡ mẫu lớn hơn nên đáng tin cậy hơn. So với long-context (không truy hồi), page-aware giữ citation accuracy cao hơn hẳn (100% so với 79,4%) trong khi chỉ tốn khoảng 10% lượng token.

---

## 4. Error Analysis

### 4.1. `test_10` (page_aware) — lỗi số liệu thật

Model trả lời "12 kho vận" thay vì đúng "2 kho vận" — nhầm lẫn với con số nhà máy (12) liền kề trong cùng đoạn. Đối chiếu với `fixed_size` (chấm partial, cũng nhầm số nhà máy) và `longcontext` (trả lời đúng cả 4 con số) cho thấy lỗi nằm ở khâu retrieval/chunking khi xử lý các con số liền kề trong cùng đoạn văn, không phải do model thiếu khả năng.

### 4.2. `test_25` (page_aware, fixed_size) — grounded nhưng phạm vi không đủ, tính là False Acceptance

Theo phản hồi của GVHD: câu trả lời có căn cứ, citation đúng, nhưng chỉ đáp ứng một phần phạm vi câu hỏi (chỉ có số liệu 12 ngày đêm, không có tổng cả 2 lần). Đánh giá tách 2 trục độc lập: **answer correctness = partial** (không phải "wrong" vì không bịa số liệu), đồng thời **vẫn tính là False Acceptance** trong chỉ số abstention vì đáng lẽ phải từ chối theo đúng phạm vi câu hỏi. Không kết luận `longcontext` có chính sách abstention tốt hơn chỉ từ 1 trường hợp — cần điều tra thêm ở tập dữ liệu held-out riêng cho việc này.

### 4.3. `test_18` (page_aware, fixed_size) — False Refusal, retrieval bỏ sót nội dung có thật

Câu có đáp án đầy đủ trong tài liệu nhưng cả 2 cấu hình RAG đều từ chối trả lời, trong khi `longcontext` trả lời đúng hoàn toàn. Nguyên nhân nhiều khả năng nằm ở bước retrieval — đoạn chứa thông tin không lọt top-k.

### 4.4. `test_34` (fixed_size) — False Refusal mới phát hiện với bộ 50 câu

Câu về chương trình "Cùng Vinamilk tích lũy điểm, vui xuân đón Tết" — có đáp án rõ ràng trong tài liệu (trang 34), `page_aware` trả lời đúng nhưng `fixed_size` từ chối. Đây là 1 trong các nguyên nhân khiến `false_refusal_rate` của `fixed_size` tăng từ 5,9% (25 câu) lên 11,8% (50 câu) — cho thấy fixed-size chunking có xu hướng mất thông tin nhiều hơn page-aware khi cỡ mẫu lớn hơn, củng cố thêm luận điểm chọn `page_aware` làm cấu hình chính thức.

### 4.5. `test_20` (fixed_size) — lỗi rớt dấu thanh trong lớp text layer của PDF gốc

Câu trả lời đúng ý chính nhưng tự thêm câu mâu thuẫn với đáp án mẫu. Đã xác nhận nguyên nhân bằng đối chứng độc lập giữa 2 công cụ đọc PDF khác nhau (`pdfplumber` và `PyMuPDF`) — cả 2 cho kết quả rớt dấu thanh giống hệt nhau tại các trang liên quan, xác nhận đây là lỗi trong lớp text layer nhúng của file PDF gốc (`normal_lichsudang_C1&2_60tr.pdf`), không phải lỗi logic hệ thống. Đã thử khắc phục bằng OCR trên bản held-out (không đụng corpus chính thức) — kết quả: loại bỏ được lỗi rớt dấu nhưng phát sinh lỗi khác (nhầm ký tự gần giống: "phản đế"→"phản đề"), retrieval cho thấy dấu hiệu tán loạn trích dẫn hơn. Kết luận: OCR đơn giản không phải giải pháp triệt để, cần hướng khác cho phiên bản sau.

---

## 5. Giới hạn kỹ thuật đã biết

| Giới hạn | Trạng thái |
|---|---|
| File VNI — một số chuyển đổi còn lỗi có quy luật | Đã xác nhận, ghi nhận, không sửa trong bản khóa hiện tại |
| File mixed-scan | **Đã đóng** — xác nhận qua kiểm tra trực tiếp nhiều trang (cả 3 trạng thái SCAN/EMPTY/TEXT): hệ thống phân loại và xử lý đúng thiết kế |
| Lỗi rớt dấu thanh trong text layer PDF (file Lịch sử Đảng) | Đã xác nhận nguyên nhân (mục 4.5), lan rộng hơn 1 trang ban đầu tưởng, chưa có giải pháp triệt để |
| File mất dấu hoàn toàn | Gây fail hoàn toàn ở bước retrieval. **Đã bổ sung Tuần 8:** cảnh báo trên giao diện Streamlit khi phát hiện |
| PDF lớn | Giới hạn an toàn 60 trang/file (dựa trên stress-test thực nghiệm Tuần 1) |

---

## 6. Kế hoạch Tuần 8

1. Hoàn thiện hồ sơ cuối kỳ: báo cáo tổng hợp, README, kịch bản video demo.
2. Không thực hiện thêm bất kỳ thay đổi nào lên `config.py`, `tau`, `k`, chunk size, prompt dựa trên kết quả Test Set 50 câu vừa có — giữ nguyên trạng thái đã đánh giá, không dùng để tinh chỉnh thêm cho phiên bản hiện tại.
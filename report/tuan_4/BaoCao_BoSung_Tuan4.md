# BÁO CÁO BỔ SUNG — Sau Tuần 4: Chẩn đoán & Khắc phục lỗi Retrieval/Chunking

**Vị trí dự kiến:** Phụ lục của báo cáo Tuần 4, hoặc mở đầu báo cáo Tuần 5
(gộp khi hoàn thiện Tuần 5 — xem ghi chú cuối file).

**Bối cảnh:** Sau khi báo cáo Tuần 4 chính (khóa Kịch bản B, tau=0.38) đã
hoàn thành, thầy yêu cầu làm rõ nguyên nhân các câu answerable bị trả lời
sai/thiếu bằng oracle-context test. Phần việc dưới đây phát sinh từ yêu cầu
đó, thực hiện sau khi báo cáo Tuần 4 đã gửi.

---

## 1. Oracle-Context Test — xác định nguyên nhân gốc

**Phương pháp:** `script/pilot_tuan4/oracle_context_test.py` — gọi thẳng
`qa_generator.generate_answer()` thật (đúng prompt/model/retry của hệ thống
production), chỉ thay khâu retrieval bằng cách nhét trực tiếp đúng trang có
đáp án (score=1.0, tau=0.0), bỏ qua hoàn toàn bước tìm kiếm tự động.

**Logic chẩn đoán:** Nếu model vẫn từ chối dù có đúng ngữ cảnh → lỗi
generation (lỗi model). Nếu model trả lời đúng → lỗi nằm ở retrieval/chunking
(hệ thống không tìm/tách đúng đoạn), không phải lỗi model.

**Kết quả: 8/8 câu nghi vấn, model đều trả lời được khi có đúng ngữ cảnh** —
không câu nào bị model tự chối. Kết luận: **100% nguyên nhân nằm ở tầng
Retrieval/Chunking của hệ thống, không phải lỗi model, không phải dev set
sai, không phải tài liệu thiếu thông tin** (dev set đã được xác nhận đúng
bằng nguyên văn PDF + ảnh chụp trực tiếp, không dựa vào text đã qua xử lý).

**Phân loại nguyên nhân gốc theo từng câu:**

| Câu | Nguyên nhân gốc | Bằng chứng |
|---|---|---|
| dev_05, dev_06 | Lỗi retrieval (không tìm đúng trang) | Trang 7/38 có đáp án nhưng không lọt vào top-k |
| dev_19, dev_25 | Lỗi retrieval (bỏ sót 1 phần trang liên quan) | Trang 29 (dev_19), câu chứa ngày 13-3-1954 (dev_25) không được lấy |
| dev_10, dev_23 | Lỗi chunking cắt cụt (đúng trang, thiếu mảnh) | Trang dài bị chia nhỏ, mảnh chứa nội dung cần thiết không được xếp hạng đủ cao |
| dev_16, dev_18 | Lỗi trích xuất PDF (bảng/biểu đồ bị xáo trộn thứ tự) | Xác nhận bằng ảnh chụp trực tiếp: tài liệu có đủ thông tin, nhưng PDF→text làm lẫn lộn nhãn-số liệu |

---

## 2. Thử nghiệm khắc phục — Lần 1: Tăng k (3→5), giữ nguyên chunk 256 token

**Kết quả: KHÔNG hiệu quả.** dev_10, dev_23 không cải thiện (nội dung câu
trả lời gần như y hệt). Còn gây **tác dụng phụ**: dev_25 từ "trả lời được
(dù thiếu)" chuyển thành "từ chối hẳn" — khả năng do thêm ngữ cảnh khiến
model kém tự tin hơn, không phải tăng chất lượng.

→ **Đã loại bỏ hướng này**, không dùng làm cấu hình chính thức. Dữ liệu giữ
lại làm minh chứng: `results/tuan4_pilot/dev_qa_results_FINAL_locked.csv`.

## 3. Thử nghiệm khắc phục — Lần 2: Tăng chunk (256→320) + gửi kèm ảnh trang biểu đồ

**Script:** `script/pilot_tuan4/run_fixes_verification.py`.

**Phát hiện phụ quan trọng:** heuristic tự động nhận diện "trang có biểu đồ"
(dựa trên số lượng object vector ≥ 40) **không dùng được** — 44/53 trang tài
liệu Vinamilk đều vượt ngưỡng do cả tài liệu dùng phong cách infographic
(viền/icon trang trí ở mọi trang). Trang 7 (biết chắc cần ảnh) chỉ đạt 179 —
thấp hơn nhiều trang chữ bình thường khác. → Chuyển sang danh sách thủ công
`CHART_HEAVY_PAGES_MANUAL = {("normal_vinamilkbaocao2014_53tr.pdf", 7)}`.

**Kết quả thử k=5 + chunk=320 + ảnh trang 7:**
- ✅ dev_05: sửa được (nhờ ảnh) — "35.704 tỷ đồng" đúng.
- ✅ dev_06: sửa được (nhờ chunk 320, không cần ảnh).
- ❌ dev_16: đủ 3 số nhưng vẫn gán nhầm nhãn.
- ❌ dev_18: vẫn thiếu 1-2/4 tiểu ban.
- ❌ dev_19: vẫn thiếu số liệu.
- 🟡 dev_10: cải thiện 1 phần, chưa đủ.
- ❌ dev_23: **tệ hơn** — chỉ còn 2/5 điểm.
- ❌ dev_25: vẫn sai.

**Kết quả thử lại k=15 (cùng chunk=320 + ảnh trang 7):**
- ✅ dev_05, dev_06: vẫn đúng.
- ❌ dev_16, dev_18: **vẫn KHÔNG sửa được** dù tăng k — xác nhận đây là vấn
  đề trích xuất PDF (bảng/biểu đồ), không phải vấn đề thiếu ngữ cảnh.
- ✅ dev_19: sửa được — đủ cả 2 số (Sản xuất 3.860.330 / Chăn nuôi 387.125).
- ✅ dev_10: gần đủ — có đầy đủ phần "Về chính trị".
- 🟡 dev_23: về lại 3/5 điểm (bằng mức cũ).
- ✅ dev_25: sửa được — đúng cả 2 ngày.

## 4. Quyết định chốt chính thức

**Chốt `k=15` và `MAX_TOKENS_PER_CHUNK=320`** làm cấu hình chính thức trong
`chunker.py` và `run_dev_qa.py` (mặc định mới). Lý do: cải thiện rõ rệt
(4/8 câu sửa hoàn toàn/gần hoàn toàn), không có tác dụng phụ mới phát sinh so
với bản k=3 gốc.

**Ghi nhận `dev_16`, `dev_18` là giới hạn đã biết** (known limitation) — đã
thử 2 cách (tăng ngữ cảnh, gửi kèm ảnh) nhưng chưa giải quyết triệt để.
Nguyên nhân gốc là cách PDF chuyển thành văn bản làm xáo trộn thứ tự
nhãn-số liệu trong bảng/biểu đồ phức tạp — cần hướng khác hẳn (OCR bảng
chuyên dụng, tăng độ phân giải ảnh, hoặc chấp nhận giới hạn) cho công việc
sau này.

**Tính năng gửi kèm ảnh (dev_05) đã kiểm chứng thành công nhưng CHƯA tích
hợp vào pipeline chính thức** (`qa_generator.py`) — mới chỉ chạy trong script
thử nghiệm riêng `run_fixes_verification.py`. Việc tích hợp chính thức để
lại cho Tuần 5 nếu còn thời gian.

## 5. Kết quả cuối cùng sau khi chốt (68 câu, Kịch bản B, k=15, chunk=320)

| Chỉ số | page_aware | fixed_size |
|---|---|---|
| Answerable trả lời được | 22/23 | 22/23 |
| Đúng hoàn toàn | 20/23 (87%) | 17/23 (74%) |
| Đúng một phần | 1/23 | 4/23 |
| Sai | 2/23 (dev_05, dev_16) | 2/23 (dev_05, dev_16) |
| Unanswerable — chặn đúng / lọt hẳn | 11/11 / 0 | 11/11 / 0 |
| Citation accuracy | 22/22 = 100% | 21/22 = 95.5% |

**So với bản gốc Tuần 4 (k=3):** page_aware answerable trả lời được từ
16/23 → 22/23; citation accuracy từ 93.8% → 100%. Cải thiện rõ rệt, xác nhận
hướng sửa (k=15 + chunk=320) đúng đắn.

Dữ liệu chính thức: `results/tuan4_pilot/dev_qa_results_FINAL_v3.csv`.

---

*Ghi chú: file này sẽ được gộp vào báo cáo Tuần 5 (phần mở đầu, trước khi
trình bày công việc Tuần 5 chính thức) khi hoàn thiện Tuần 5, hoặc đính kèm
làm phụ lục báo cáo Tuần 4 nếu thầy muốn xem trước.*
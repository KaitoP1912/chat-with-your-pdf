# Log kiểm thử toàn bộ Corpus qua `app_cli.py` — Việc B, Tuần 6

**Trạng thái:** ✅ Đã có kết quả với câu hỏi cụ thể cho cả 8/8 file. 6/8 file
kết luận được rõ ràng. **2 file (6, 8) vẫn abstain dù đã đổi câu hỏi cụ thể
— đây là phát hiện thật cần điều tra thêm**, không còn là do câu hỏi dở
như nghi ngờ ban đầu (xem mục 3 và mục 4).

---

## 1. Hai phát hiện quan trọng từ lần chạy đầu (cần bạn xử lý/ghi nhận)

### 1.1. Bug môi trường: `--vncorenlp_dir` mặc định là đường dẫn tương đối

Lần chạy đầu tiên (không truyền `--vncorenlp_dir`), **7/8 file** đều lỗi
`java.lang.NoClassDefFoundError: vn/pipeline/VnCoreNLP`. Nguyên nhân: mặc
định của `script/app_cli.py` là đường dẫn **tương đối** (`./vncorenlp_models`),
nhưng `py_vncorenlp` bắt buộc đường dẫn tuyệt đối — đúng như báo cáo Tuần 3
của bạn đã ghi nhận và khắc phục cho các script khác, nhưng có vẻ
**chưa được áp dụng trong `app_cli.py`**. Sau khi truyền cờ
`--vncorenlp_dir "<đường_dẫn_tuyệt_đối>"` bằng tay, toàn bộ 8 file chạy qua
được Bước 3 (dựng index) không lỗi.

⚠️ **Điều này mâu thuẫn với báo cáo Tuần 5** ("CLI end-to-end: đã chạy thật,
hoạt động đúng"). Bạn nên tự kiểm tra: lúc đó có phải đã luôn truyền
`--vncorenlp_dir` tuyệt đối bằng tay (không dùng default) mà không ghi chú
lại trong báo cáo? Không thuộc phạm vi Việc B để tự sửa code, nhưng nên ghi
vào `SYSTEM_LOCK.md` (Việc C) như một điểm cần lưu ý khi chạy Test Set
chính thức Tuần 7 — nếu không, người chạy lại (kể cả bạn sau này) dễ gặp
lại lỗi này mà tưởng là bug mới.

### 1.2. Phát hiện kiến trúc: `app_cli.py` không áp dụng chính sách từ chối FULL_SCAN đã khóa Tuần 1-2

Test file `scan_nd238_14tr.pdf` (full-scan, kỳ vọng "bị từ chối rõ ràng")
cho kết quả: đọc PDF thành công (không lỗi) → tạo 0 chunk → dựng index 0
chunk → **abstain với lý do `retrieval_threshold`** — **in ra y hệt** một
câu hỏi ngoài phạm vi tài liệu bình thường (`"Không tìm thấy thông tin liên
quan trong tài liệu."`), không có bất kỳ dấu hiệu nào cho biết file đã bị
từ chối vì là FULL_SCAN.

**Nguyên nhân gốc:** `source/retrieval/ingest_glue.py:build_clean_pages()`
— hàm mà `app_cli.py` (và cả pipeline RAG nói chung) dùng để đọc PDF — chỉ
gọi `load_pdf_pages()` + `normalize_page_text()`, **không hề gọi**
`scan_detector.detect_scan()` / `should_reject()`. Chính sách "từ chối
FULL_SCAN" đã khóa từ Tuần 1-2 (`scan_detector.py`) tồn tại trong code
nhưng **không được nối vào luồng `build_clean_pages` → `app_cli.py`**.

So sánh: bản `script/app_ui.py` (giao diện, refactor từ `app_streamlit.py`)
**có** tự gọi `detect_scan`/`should_reject` riêng trước khi build index, nên
UI báo lỗi rõ ràng "File là PDF scan hoàn toàn..." — còn CLI thì không.

**Tại sao đây là vấn đề đáng lưu ý (không phải chỉ là chi tiết vụn):**
người dùng `app_cli.py` không thể phân biệt được 2 tình huống rất khác
nhau — "tài liệu không đọc được (scan)" vs "câu hỏi thực sự ngoài phạm vi
tài liệu" — vì cả 2 đều trả về đúng 1 câu abstain giống hệt nhau. Điều này
ảnh hưởng trực tiếp tới độ tin cậy của Test Set Tuần 7 nếu Test Set có câu
hỏi trên tài liệu scan mà không được ghi chú rõ nguyên nhân abstain.

→ **Đề xuất (để bạn quyết định, ngoài phạm vi sửa code của Việc B):** khi
làm `SYSTEM_LOCK.md` hoặc trước Tuần 7, nên xem xét thêm bước gọi
`scan_detector` vào `build_clean_pages()` hoặc trực tiếp vào `app_cli.py`,
để hành vi CLI nhất quán với UI.

---

## 2. Vấn đề với câu hỏi thăm dò ban đầu (lỗi của mình, không phải lỗi hệ thống)

4 file thăm dò (4, 6, 7, 8) đều dùng câu hỏi dạng tóm tắt toàn văn bản
("...quy định về vấn đề gì?", "Luật này có tên gọi đầy đủ là gì?") và **cả
4 đều abstain** với lý do `retrieval_threshold`. Vì hệ thống retrieval so
khớp cosine similarity theo **từng chunk cụ thể**, câu hỏi tóm tắt quá
chung chung nhiều khả năng không đủ giống bất kỳ chunk nào để vượt
`tau=0.38` — đây nhiều khả năng là **lỗi chọn câu hỏi của mình**, không
phải bằng chứng lỗi `text_normalizer`. Hệ quả: mục tiêu ban đầu (kiểm tra
dấu tiếng Việt/mã hóa được chuẩn hóa đúng chưa) **chưa đạt được** vì chưa
có câu trả lời nào để soi xét.

Đã tra cứu nội dung thật của 2 luật để đề xuất câu hỏi cụ thể hơn (không
bịa — có nguồn):

- **Luật 118/2025/QH15** (file 6, 8): Luật sửa đổi, bổ sung 10 luật về an
  ninh, trật tự; hiệu lực thi hành từ **01/7/2026**.
  (Nguồn: luatvietnam.vn, bocongan.gov.vn)
- **Luật 36/2024/QH15** (file 7): Luật Trật tự, an toàn giao thông đường
  bộ; gồm **9 chương, 89 điều**; hiệu lực thi hành từ **01/01/2025**.
  (Nguồn: laichau.gov.vn — "Giới thiệu Luật...")

### Câu lệnh đề nghị chạy lại (nhớ giữ cờ `--vncorenlp_dir` tuyệt đối)

```powershell
# 4. mixedscan — câu hỏi cụ thể hơn thay vì tóm tắt chung chung
python script/app_cli.py --pdf "data/corpus/mixedscan_qcvn06_38tr.pdf" --question "Phạm vi điều chỉnh của quy chuẩn kỹ thuật này là gì?" --vncorenlp_dir "D:\PHUOC\HK9\TTTN\chat-with-your-pdf\vncorenlp_models"

# 6. VNI — câu hỏi có đáp án cụ thể tra được (hiệu lực từ 01/7/2026)
python script/app_cli.py --pdf "data/corpus/oldenc_vni_118-2025-qh15_23tr.pdf" --question "Luật này có hiệu lực thi hành từ ngày nào?" --vncorenlp_dir "D:\PHUOC\HK9\TTTN\chat-with-your-pdf\vncorenlp_models"

# 7. TCVN3 — câu hỏi có đáp án cụ thể tra được (9 chương, 89 điều / hiệu lực 01/01/2025)
python script/app_cli.py --pdf "data/corpus/oldenc_tcvn3_36-2024-qh15_53tr.pdf" --question "Luật này có hiệu lực thi hành từ ngày nào?" --vncorenlp_dir "D:\PHUOC\HK9\TTTN\chat-with-your-pdf\vncorenlp_models"

# 8. nodiacritic — cùng câu hỏi với file 6 (cùng nguồn Luật 118/2025/QH15), để so sánh chéo ảnh hưởng của việc mất dấu
python script/app_cli.py --pdf "data/corpus/nodiacritic_118-2025-qh15_20tr.pdf" --question "Luật này có hiệu lực thi hành từ ngày nào?" --vncorenlp_dir "D:\PHUOC\HK9\TTTN\chat-with-your-pdf\vncorenlp_models"
```

Nếu vẫn abstain sau khi đổi câu hỏi cụ thể hơn, lúc đó mới nên nghi ngờ đây
là lỗi thật của `text_normalizer`/retrieval trên các file mã hóa cũ/mất
dấu, thay vì lỗi chọn câu hỏi.

---

## 3. Kết quả (đã có 8/8 — 3 file cần chạy lại xác nhận, xem cột "Trạng thái kết luận")

### normal_hienphap_33tr.pdf
- Loại: chuẩn
- Câu hỏi thử: "Nhiệm kỳ của mỗi khóa Quốc hội được quy định bao nhiêu năm?"
- Kết quả: chạy được
- Ghi chú: Trả lời đúng — *"Dựa trên tài liệu, nhiệm kỳ của mỗi khóa Quốc
  hội là năm năm."* — khớp chính xác đáp án mẫu dev_03. Trích dẫn đúng
  trang 17 (đáp án mẫu), kèm 2 bridge liên quan (16-17, 20-21) và trang 20.
  1.164s, 1697 token.
- **Trạng thái kết luận: ✅ ĐẠT**

### normal_vinamilkbaocao2014_53tr.pdf
- Loại: chuẩn
- Câu hỏi thử: "Doanh thu thuần của Vinamilk năm 2014 đạt bao nhiêu tỷ đồng?"
- Kết quả: chạy được
- Ghi chú: Trả lời đúng — *"...doanh thu (hoặc doanh thu thuần) của
  Vinamilk năm 2014 đạt 35.704 tỷ đồng."* — khớp chính xác đáp án mẫu
  dev_05. **Xác nhận tính năng gửi ảnh trang biểu đồ hoạt động đúng**: log
  hiện rõ "Ảnh trang đã gửi kèm Gemini: ...#trang7" (đúng
  `CHART_HEAVY_PAGES_MANUAL`). Trích dẫn có trang 7 đúng, nhưng đi kèm **13
  trang khác** (nhiều bridge nội bộ/liên trang) — danh sách trích dẫn khá
  dài/nhiễu so với 2 file kia, đáng ghi chú (không phải lỗi, nhưng nên theo
  dõi citation accuracy trên file này ở Tuần 7 vì tau=0.38 có vẻ để lọt khá
  nhiều chunk trên file 53 trang nhiều bridge). 5.607s, 7880 token.
- **Trạng thái kết luận: ✅ ĐẠT** (kèm 1 điểm cần theo dõi: số lượng trích dẫn nhiều)

### normal_lichsudang_C1&2_60tr.pdf
- Loại: chuẩn
- Câu hỏi thử: "Hội nghị thành lập Đảng Cộng sản Việt Nam diễn ra vào thời gian nào và ở đâu?"
- Kết quả: chạy được
- Ghi chú: Trả lời đúng — *"...họp từ ngày 6-1 đến ngày 7-2-1930 tại Hương
  Cảng, Trung Quốc."* — khớp chính xác đáp án mẫu dev_09, đúng trang 14.
  1.485s, 3950 token.
- **Trạng thái kết luận: ✅ ĐẠT**

### mixedscan_qcvn06_38tr.pdf
- Loại: mixed-scan (trang 1, 38 scan theo Tuần 2)
- Câu hỏi thử (lần 2): "Phạm vi điều chỉnh của quy chuẩn kỹ thuật này là gì?"
- Kết quả: chạy được, abstain nhưng lý do đổi khác — `model_refusal` (không
  còn là `retrieval_threshold` như câu hỏi lần 1)
- Ghi chú: **Khác biệt quan trọng so với lần 1**: reason đổi từ
  `retrieval_threshold` sang `model_refusal` nghĩa là lần này CÓ chunk vượt
  ngưỡng tau=0.38 (retrieval tìm được nội dung liên quan), nhưng chính
  Gemini quyết định không đủ căn cứ để trả lời — không phải hệ thống
  "không tìm thấy gì" như trước. Cách giải thích hợp lý nhất: "Phạm vi điều
  chỉnh" của quy chuẩn thường nằm ở Điều 1, mà theo Tuần 2, **trang 1 của
  chính file này bị scan** (không có text layer, và do phát hiện ở mục 1.2,
  `app_cli.py` không loại trang scan ra — trang 1 vẫn được đưa vào pipeline
  nhưng với text rỗng). Nếu đúng vậy, đây là hành vi **ĐÚNG** (model từ
  chối đúng vì thông tin thật sự không có trong corpus text do trang 1 bị
  scan) — nhưng đây là suy luận, CHƯA xác nhận trực tiếp bằng cách xem nội
  dung trang 1 gốc. Muốn xác nhận chắc chắn, dùng
  `script/pilot_tuan2/xem_noi_dung_trang.py "data/corpus/mixedscan_qcvn06_38tr.pdf" 1 1`
  để xem trang 1 có đúng chứa "Phạm vi điều chỉnh" hay không.
- **Trạng thái kết luận: 🟢 CÓ THỂ ĐÚNG THIẾT KẾ (cần 1 bước xác nhận thêm, không phải lỗi)**

### scan_nd238_14tr.pdf
- Loại: full-scan
- Câu hỏi thử: "Nghị định này quy định về vấn đề gì?"
- Kết quả: chạy được nhưng **KHÔNG bị từ chối rõ ràng như thiết kế** — xem
  phân tích chi tiết ở mục 1.2
- Ghi chú: Hệ thống đọc file "thành công" (0 lỗi), tạo 0 chunk, rồi abstain
  với thông báo y hệt câu hỏi ngoài phạm vi thông thường — không có thông
  báo "file bị từ chối vì FULL_SCAN". Không crash, nhưng **không đúng hành
  vi đã thiết kế/khóa ở Tuần 1-2** (chính sách từ chối rõ ràng tài liệu
  FULL_SCAN không được áp dụng trong luồng `app_cli.py`).
- **Trạng thái kết luận: ⚠️ PHÁT HIỆN KIẾN TRÚC — không phải crash, nhưng
  không đúng thiết kế đã khóa. Xem mục 1.2.**

### oldenc_vni_118-2025-qh15_23tr.pdf
- Loại: mã hóa cũ (VNI Windows)
- Câu hỏi thử (lần 2): "Luật này có hiệu lực thi hành từ ngày nào?"
- Kết quả: chạy được nhưng **vẫn abstain** (`retrieval_threshold`)
- Ghi chú: ⚠️ **Đáng ngờ thật sự** — không còn là do câu hỏi dở nữa (đã đổi
  sang câu hỏi cụ thể, có đáp án tra được rõ ràng: 01/7/2026), và file 7
  (cũng là mã hóa cũ, cùng dạng câu hỏi "hiệu lực thi hành từ ngày nào?")
  **trả lời đúng hoàn toàn**. Sự khác biệt này gợi ý vấn đề nằm riêng ở
  file 6 — nhiều khả năng là chất lượng chuyển đổi **VNI → Unicode** kém
  hơn TCVN3 → Unicode cho tài liệu cụ thể này (candidate ranking có thể đã
  chấm điểm dưới ngưỡng 0.50, giữ nguyên bản gốc theo "Zero-touch policy",
  khiến đoạn văn bản vẫn còn ký tự VNI thô → model embedding tiếng Việt
  không hiểu được → cosine similarity thấp, không vượt tau). **Đây là giả
  thuyết, chưa xác nhận.** Để xác nhận, cần soi trực tiếp text đã chuẩn hóa
  của file này bằng:
  `python script/pilot_tuan2/check_all.py "data/corpus/oldenc_vni_118-2025-qh15_23tr.pdf" "results/full_text_inspect/vni_recheck.txt"`
  rồi đọc file .txt xem còn ký tự VNI lạ hay không, đặc biệt ở trang chứa
  "Điều khoản thi hành".
- **Trạng thái kết luận: 🔴 CẦN ĐIỀU TRA THÊM — nghi vấn thật, không tự kết luận vội**

### oldenc_tcvn3_36-2024-qh15_53tr.pdf
- Loại: mã hóa cũ (TCVN3/ABC)
- Câu hỏi thử (lần 2): "Luật này có hiệu lực thi hành từ ngày nào?"
- Kết quả: chạy được, **trả lời đúng**
- Ghi chú: Trả lời: *"...có hiệu lực thi hành từ ngày 01 tháng 01 năm 2025
  (trừ trường hợp quy định tại khoản 2 Điều 88 có hiệu lực từ ngày 01
  tháng 01 năm 2026)."* — khớp đúng mốc chính 01/01/2025 mà mình tra được
  từ nguồn ngoài (laichau.gov.vn). Phần chi tiết bổ sung về khoản 2 Điều 88
  không nằm trong dữ liệu mình tra cứu được nên chưa xác minh chéo, nhưng
  không có dấu hiệu bịa (đúng định dạng ngày tháng, đúng cấu trúc luật VN).
  **Xác nhận: chuẩn hóa TCVN3 → Unicode hoạt động đúng, đủ tốt để retrieval
  và generation ra câu trả lời chính xác.** 1.4s, 1579 token.
- **Trạng thái kết luận: ✅ ĐẠT**

### nodiacritic_118-2025-qh15_20tr.pdf
- Loại: mất dấu
- Câu hỏi thử (lần 2): "Luật này có hiệu lực thi hành từ ngày nào?"
- Kết quả: chạy được nhưng **vẫn abstain** (`retrieval_threshold`)
- Ghi chú: Cùng nguồn Luật 118/2025/QH15 với file 6, cùng loại kết quả
  (abstain retrieval_threshold) — nhưng khác với file 6, trường hợp này
  **có căn cứ đã biết trước** để giải thích: báo cáo Tuần 2 xác nhận
  `text_normalizer` chỉ **phát hiện** mất dấu (gắn cờ
  `likely_missing_diacritics=True`), không có bằng chứng đã **khôi phục**
  dấu. Nếu văn bản vẫn hoàn toàn không dấu ("Luat nay co hieu luc..."),
  model embedding tiếng Việt (huấn luyện trên văn bản có dấu) rất khó cho
  điểm cosine cao với câu hỏi có dấu đầy đủ → dễ hiểu vì sao rớt ngưỡng
  tau. **Đây là giới hạn đã biết từ Tuần 2, không phải lỗi mới phát sinh**
  — nhưng đáng lưu ý: giới hạn này ảnh hưởng tới khả năng TRẢ LỜI ĐƯỢC câu
  hỏi chứ không chỉ ảnh hưởng chính tả câu trả lời như báo cáo Tuần 2 có
  thể ngụ ý.
- **Trạng thái kết luận: 🟡 XÁC NHẬN GIỚI HẠN ĐÃ BIẾT (Tuần 2), mức độ ảnh
  hưởng nghiêm trọng hơn dự kiến — nên ghi rõ vào SYSTEM_LOCK.md**

### word_118-2025-qh15_28tr.docx (tùy chọn, ngoài 8 file bắt buộc)
- Chưa test.

---

## 4. Tóm tắt cuối cùng (8/8 file có kết quả thật)

| # | File | Kết luận |
|---|---|---|
| 1 | normal_hienphap_33tr.pdf | ✅ ĐẠT — trả lời đúng, khớp ground truth |
| 2 | normal_vinamilkbaocao2014_53tr.pdf | ✅ ĐẠT — trả lời đúng, ảnh trang biểu đồ hoạt động đúng |
| 3 | normal_lichsudang_C1&2_60tr.pdf | ✅ ĐẠT — trả lời đúng, khớp ground truth |
| 4 | mixedscan_qcvn06_38tr.pdf | 🟢 Có thể đúng thiết kế (model_refusal hợp lý do trang 1 scan) — cần 1 bước xác nhận thêm |
| 5 | scan_nd238_14tr.pdf | ⚠️ Phát hiện kiến trúc — không crash nhưng không đúng chính sách từ chối FULL_SCAN đã khóa (mục 1.2) |
| 6 | oldenc_vni_118-2025-qh15_23tr.pdf | 🔴 CẦN ĐIỀU TRA — nghi vấn chuẩn hóa VNI kém hơn TCVN3, chưa xác nhận |
| 7 | oldenc_tcvn3_36-2024-qh15_53tr.pdf | ✅ ĐẠT — trả lời đúng, xác nhận chuẩn hóa TCVN3 hoạt động tốt |
| 8 | nodiacritic_118-2025-qh15_20tr.pdf | 🟡 Xác nhận giới hạn đã biết (Tuần 2) — mất dấu chưa được khôi phục, ảnh hưởng tới khả năng trả lời |
| 9 | word_118-2025-qh15_28tr.docx | Chưa test (tùy chọn, không bắt buộc) |

**Số liệu:**
- ✅ Chạy đúng, trả lời đúng: **4/8** (file 1, 2, 3, 7)
- 🟢 Nghi là đúng thiết kế, cần xác nhận thêm 1 bước: **1/8** (file 4)
- 🟡 Xác nhận giới hạn đã biết, mức ảnh hưởng nghiêm trọng hơn dự kiến: **1/8** (file 8)
- 🔴 Nghi vấn thật cần điều tra kỹ hơn (không tự kết luận): **1/8** (file 6)
- ⚠️ Phát hiện kiến trúc (không phải crash, nhưng sai thiết kế đã khóa): **1/8** (file 5)
- LỖI BẤT NGỜ (crash không rõ nguyên nhân): **0/8**

**Nhận xét chung:** 3 file "chuẩn" hoạt động đúng 100%. Trong 5 file "lỗi"
(edge case), có 1 điểm sáng rõ ràng (TCVN3 hoạt động tốt) và 1 điểm cần
điều tra thật sự (VNI kém hơn hẳn TCVN3 dù cùng loại lỗi mã hóa cũ — không
nên bỏ qua chỉ vì "đã là file lỗi thì abstain là bình thường"). Đây chính
là kiểu phát hiện mà việc kiểm thử toàn bộ 9 file (thay vì chỉ 3 file dev
set) được kỳ vọng tìm ra.

## 5. Việc cần làm tiếp theo (ngoài phạm vi Việc B, để bạn quyết định)

1. **File 6 (VNI)**: chạy `check_all.py` để soi text đã chuẩn hóa, xác
   nhận có còn sót ký tự VNI thô hay không (lệnh ở mục kết quả file 6 phía
   trên).
2. **File 4 (mixedscan)**: xác nhận trang 1 gốc có đúng chứa "Phạm vi điều
   chỉnh" hay không bằng `xem_noi_dung_trang.py` (lệnh ở mục kết quả file 4).
3. **File 5 (full-scan)** và **file 8 (mất dấu)**: cả 2 phát hiện đều nên
   đưa vào `SYSTEM_LOCK.md` (Việc C) làm "giới hạn/rủi ro đã biết trước
   Tuần 7", kể cả khi chưa sửa code — để Test Set Tuần 7 không bị hiểu
   nhầm kết quả abstain trên các dạng file này là lỗi hệ thống mới.
4. **Bug `--vncorenlp_dir`** (mục 1.1): cân nhắc sửa default trong
   `app_cli.py` thành đường dẫn tuyệt đối (`os.path.abspath(...)`, giống
   cách `app_ui.py` đã làm) — việc sửa nhỏ, ít rủi ro, nên làm sớm để tránh
   người khác gặp lại lỗi này.
# BÁO CÁO TỔNG KẾT ĐỀ TÀI THỰC TẬP TỐT NGHIỆP

## Xây dựng ứng dụng hỏi đáp tài liệu PDF tiếng Việt có trích dẫn số trang bằng kỹ thuật RAG
### (Chat with Your PDF)

**Sinh viên thực hiện:** Võ Thành Phước — MSSV: 079205022977
**Giảng viên hướng dẫn:** ThS. Nguyễn Thanh Tiến
**Thời gian thực hiện:** Tuần 1 – Tuần 8
**Mã nguồn:** <https://github.com/KaitoP1912/chat-with-your-pdf>

---

## 1. Tóm tắt đề tài

Xây dựng một ứng dụng hỏi đáp tài liệu PDF tiếng Việt sử dụng kỹ thuật RAG (Retrieval-Augmented Generation), có khả năng trích dẫn đúng số trang chứa đáp án. Đề tài tập trung giải quyết câu hỏi nghiên cứu trung tâm:

> Chunking theo ranh giới trang có cải thiện độ chính xác trích dẫn số trang so với (a) chunking cố định không theo trang, và (b) không dùng truy hồi (đưa nguyên văn bản vào ngữ cảnh dài của Gemini) hay không? Cải thiện bao nhiêu phần trăm?

Phạm vi bắt buộc: PDF tiếng Việt có text layer. Hỗ trợ Word (qua LibreOffice) là phần mở rộng, không thuộc cam kết chính thức.

---

## 2. Kiến trúc hệ thống

| Trạm | Thành phần | Mô tả |
|---|---|---|
| 1 — Ingestion | `pdf_loader.py`, `scan_detector.py`, `text_normalizer.py` | Trích xuất PDF theo trang (pdfplumber), phân loại TEXT/SCAN/MIXED_SCAN/FULL_SCAN, chuẩn hóa bảng mã cũ (TCVN3/VNI→Unicode) bằng Candidate Ranking + Lexicon Scoring, khôi phục lỗi glyph "ư" rớt |
| 2 — Chunking & Indexing | `chunker.py`, `word_segmenter.py`, `vectorstore.py` | Tách từ tiếng Việt (py_vncorenlp), 2 chiến lược chunking (page-aware + bridge chunk / fixed-size baseline), embedding (bkai-foundation-models/vietnamese-bi-encoder), **Hybrid Search: FAISS (dense) + BM25 (sparse) kết hợp qua Reciprocal Rank Fusion (RRF)** |
| 3 — Generation | `qa_generator.py` | Sinh câu trả lời qua Gemini 3.5-flash-lite, citation lấy từ metadata (không cho LLM tự bịa số trang), chính sách abstention 2 tầng (ngưỡng retrieval τ + QUY TẮC 2 ở tầng generation), gửi kèm ảnh trang cho các trang biểu đồ/bảng phức tạp |
| 4 — UI | Streamlit (`app_ui.py`) | Giao diện chat, upload file, hiển thị trích dẫn dạng chip, cảnh báo trang scan và cảnh báo mất dấu tiếng Việt (bổ sung Tuần 8) |
| 5 — Evaluation | `script/pilot_tuan*/` | Đo Hit@k, citation accuracy, false acceptance/refusal, answer correctness, latency, token, bridge-case riêng |

### Cấu hình đã khóa (từ 28/8/2026, xác nhận không đổi tới hết Tuần 7)

`tau=0.38`, `k=15`, `chunk_size=320 token` (page-aware) / `170 từ` (fixed-size baseline), `bridge_words=128`, model `gemini-3.5-flash-lite`, `temperature=0.0`.

---

## 3. Dữ liệu — Corpus chính thức (khóa 31/07/2026)

9 file, gồm 3 nhóm: (1) chuẩn — dùng soạn câu hỏi dev/test chính (Hiến pháp, Vinamilk, Lịch sử Đảng); (2) edge-case — dùng sanity-check module xử lý (scan toàn phần, scan từng phần, TCVN3, VNI, mất dấu); (3) mở rộng — 1 file Word, không tính vào bộ câu hỏi chính.

Dev set: 34 câu, dùng chọn ngưỡng τ và tinh chỉnh tham số ở Tuần 4-5. Test Set: **50 câu (34 answerable gồm 4 bridge case, 16 unanswerable)**, độc lập với Dev Set, dùng cho đánh giá chính thức Tuần 7. 25 câu đầu khóa từ Tuần 5; 25 câu bổ sung soạn ở Tuần 7 theo quy trình kiểm chứng nghiêm ngặt (xem mục 4, dòng Tuần 7).

---

## 4. Tiến độ theo tuần (tóm tắt)

| Tuần | Nội dung chính |
|---|---|
| 1 | Chốt câu hỏi nghiên cứu, giới hạn vận hành (60 trang, 20MB, timeout 90s — dựa trên stress-test thực nghiệm), corpus 9 file, rubric chấm |
| 2 | Trạm 1 hoàn thiện: ingestion, scan detection, chuẩn hóa bảng mã (F1=1.000, CER=0.3%, FCR=0%, 138/138 test pass) |
| 3 | Trạm 2 hoàn thiện: page-aware chunking + bridge chunk, fixed-size baseline, embedding + FAISS, demo retrieval end-to-end thành công |
| 4 | Tích hợp Gemini QA, oracle-context test xác định 100% lỗi answerable nằm ở retrieval/chunking (không phải model), chốt k=15 + chunk=320 sau 2 vòng thử nghiệm |
| 5 | Baseline long-context, chọn ngưỡng τ=0.38 qua threshold sweep trên dev set, tích hợp Hybrid Search (FAISS+BM25 qua RRF) và gửi ảnh trang biểu đồ chính thức |
| 6 | Giao diện Streamlit, kiểm thử 8/8 file corpus hợp lệ, khóa cứng hệ thống (`SYSTEM_LOCK.md`), chạy sớm Test Set trên 3 cấu hình, vá lỗi FULL_SCAN và verify `vectorstore.py` (cả 2 xác nhận không ảnh hưởng code path Test Set) |
| 7 | Đánh giá chính thức Test Set 50 câu, chấm tay đầy đủ, xác minh lịch sử khóa hệ thống bằng bằng chứng thời gian file cụ thể, error analysis 5 trường hợp |
| 8 | Hoàn thiện hồ sơ cuối kỳ: báo cáo tổng kết, README, kịch bản video demo, cảnh báo mất dấu trên giao diện, dọn dẹp mã nguồn |

**Lưu ý minh bạch về quá trình mở rộng Test Set:** trong quá trình chuẩn bị báo cáo Tuần 7, lần thử mở rộng Test Set từ 25 lên 50 câu đầu tiên (bằng công cụ hỗ trợ AI) bị phát hiện rò rỉ dữ liệu — 25 câu "mới" khi đó thực chất là bản sao gần như nguyên văn của Dev Set. Đã khôi phục về 25 câu gốc, sau đó soạn lại 25 câu bổ sung theo quy trình kiểm chứng 3 lớp độc lập (đối chiếu vùng cấm bằng code, đối chiếu nội dung trực tiếp với PDF gốc, đọc tay xác nhận) trước khi đưa vào đánh giá chính thức — không lặp lại sai sót trước đó. Toàn bộ quá trình chấm tay cũng được rà soát lại nhiều lần theo phản hồi trực tiếp của GVHD.

---

## 5. Kết quả đánh giá chính thức (Test Set, 50 câu)

| Chỉ số | page_aware (đề xuất) | fixed_size (baseline) | longcontext (baseline) |
|---|---|---|---|
| Hit@3 (retriever) | **94,1%** | 88,2% | n/a |
| Citation accuracy | **100,0%** | 90,0% | 79,4% |
| False acceptance rate | 6,2% | 6,2% | **0,0%** |
| False refusal rate | 2,9% | 11,8% | **0,0%** |
| Answer correctness (full/partial/wrong) | 82,4% / 14,7% / 2,9% | 74,2% / 25,8% / 0,0% | 88,2% / 11,8% / 0,0% |
| Latency p50 | 1,78s | **1,40s** | 1,29s |
| Token trung bình | 4.197 | **2.711** | 41.417 |
| Bridge-case Hit@3 | **100% (4/4)** | 100% (4/4) | n/a |

### Trả lời câu hỏi nghiên cứu trung tâm

Page-aware chunking **cải thiện Hit@3 khoảng 5,9 điểm phần trăm** (94,1% so với 88,2%) và **cải thiện citation accuracy khoảng 10 điểm phần trăm** (100% so với 90,0%) so với fixed-size chunking, trên bộ Test Set 50 câu độc lập. So với việc không dùng truy hồi (long-context), page-aware giữ được citation accuracy cao hơn hẳn (100% so với 79,4%) trong khi chỉ tốn khoảng 10% lượng token — đánh đổi hợp lý cho triển khai thực tế.

Kết quả Hit@3 và citation accuracy giữ nguyên gần như y hệt so với lần đánh giá sơ bộ 25 câu đầu — củng cố độ tin cậy của kết luận, không phải hiệu ứng ngẫu nhiên của cỡ mẫu nhỏ. `false_refusal_rate` của `fixed_size` tăng từ 5,9% lên 11,8% khi mở rộng mẫu — cho thấy chiến lược fixed-size có xu hướng mất thông tin nhiều hơn khi cỡ mẫu lớn hơn, càng củng cố lựa chọn `page_aware` làm cấu hình chính thức.

---

## 6. Error Analysis (chi tiết xem Báo cáo Tuần 7)

5 trường hợp đáng chú ý đã phân tích: (1) `test_10` — lỗi số liệu do nhầm lẫn giữa các con số liền kề trong retrieval/chunking; (2) `test_25` — câu trả lời grounded, citation đúng, nhưng vượt phạm vi tài liệu, tính là False Acceptance theo đúng chính sách (phản hồi GVHD); (3) `test_18` — False Refusal, retrieval bỏ sót nội dung có thật dù model có khả năng trả lời đúng khi có đủ ngữ cảnh; (4) `test_34` — False Refusal mới phát hiện ở `fixed_size` khi mở rộng mẫu 50 câu; (5) `test_20` — lỗi rớt dấu thanh trong lớp text layer PDF gốc, đã xác nhận bằng đối chứng độc lập 2 công cụ đọc PDF khác nhau (pdfplumber, PyMuPDF), đã thử OCR khắc phục nhưng chưa triệt để.

---

## 7. Giới hạn kỹ thuật đã biết

| Giới hạn | Trạng thái |
|---|---|
| File VNI — một số chuyển đổi còn lỗi có quy luật | Đã xác nhận, ghi nhận, không sửa trong bản khóa hiện tại |
| File mixed-scan | **Đã đóng** — xác nhận hoạt động đúng thiết kế qua kiểm tra trực tiếp nhiều trang |
| Lỗi rớt dấu thanh trong text layer PDF (file Lịch sử Đảng) | Đã xác nhận nguyên nhân, lan rộng hơn 1 trang ban đầu tưởng, đã thử OCR nhưng chưa triệt để |
| File mất dấu tiếng Việt hoàn toàn | Gây fail hoàn toàn retrieval. Đã bổ sung cảnh báo trên giao diện Streamlit (Tuần 8) |
| PDF lớn — latency tăng tuyến tính | Có giới hạn vận hành (60 trang/file), không phải lỗi |
| Word/.docx | Ngoài phạm vi cam kết chính thức, chưa triển khai |

---

## 8. Đóng góp và kết luận

Đề tài đã xây dựng hoàn chỉnh một pipeline RAG tiếng Việt từ ingestion đến generation, với các điểm kỹ thuật đáng chú ý: (1) cơ chế chuẩn hóa bảng mã cũ (TCVN3/VNI) dựa trên Candidate Ranking + Lexicon Scoring thay vì heuristic vá cứng, đạt F1=1.000 trên bộ benchmark 67 mẫu độc lập; (2) bridge chunk cho nội dung vắt qua ranh giới trang, đánh giá tách riêng và xác nhận Hit@3=100% trên 4 bridge case trong Test Set; (3) Hybrid Search (Dense + BM25 qua RRF) kết hợp abstention 2 tầng.

Kết quả đánh giá chính thức trên 50 câu độc lập cho thấy page-aware chunking cải thiện rõ rệt cả độ chính xác truy hồi lẫn độ chính xác trích dẫn so với fixed-size chunking, trả lời được câu hỏi nghiên cứu trung tâm của đề tài với độ tin cậy cao hơn so với đánh giá sơ bộ 25 câu.

Toàn bộ quy trình thực nghiệm tuân thủ nghiêm ngặt nguyên tắc tách biệt dev/test: tham số được khóa dựa trên dev set trước khi quan sát test set, có bằng chứng thời gian cụ thể xác nhận trình tự này, và các rủi ro phương pháp luận phát sinh trong quá trình thực hiện (rò rỉ dữ liệu khi mở rộng test set, lỗi chấm điểm tự động) đều được phát hiện và khắc phục minh bạch trước khi công bố kết quả chính thức.

---

## 9. Hướng phát triển tiếp theo

1. Điều tra và khắc phục lỗi VNI, lỗi rớt dấu thanh trong text layer PDF (đã thử OCR, cần hướng khác triệt để hơn — mở rộng cơ chế Candidate Ranking sang riêng lỗi này).
2. Xây dựng cơ chế xử lý bảng/biểu đồ phức tạp trong PDF (nguyên nhân gốc của các lỗi số liệu như `test_10`, `dev_16`, `dev_18`).
3. Điều tra sâu hơn nguyên nhân False Acceptance ở `test_25` (retrieval, lượng evidence, hay chính sách QUY TẮC 2) trên tập held-out riêng.
4. Hoàn thiện hỗ trợ Word/.docx nếu mở rộng phạm vi đề tài.
5. Cơ chế fallback cho văn bản mất dấu tiếng Việt (hiện chỉ dừng ở mức cảnh báo).
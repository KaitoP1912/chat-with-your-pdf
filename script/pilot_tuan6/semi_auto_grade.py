#!/usr/bin/env python3
"""
Semi-auto grading: chấm answer_correctness dựa trên heuristic đơn giản.
Sau đó người dùng chỉ cần kiểm tra lại những câu "uncertain" —
NHƯNG với những dòng "wrong"/"partial" vẫn nên tự đọc lại, vì heuristic
so sánh chuỗi ký tự vẫn có thể sai với câu trả lời dài/diễn đạt khác.

BẢN SỬA (so với bản gốc):
- Loại bỏ markdown (**, -, #, xuống dòng thừa, tiền tố "Dựa trên tài liệu...")
  khỏi answer_text trước khi so sánh với answer_reference (văn xuôi thuần).
  Bản gốc so sánh trực tiếp 2 chuỗi có định dạng khác nhau, khiến câu trả lời
  ĐÚNG NỘI DUNG nhưng có markdown bị chấm nhầm thành "wrong" (đã xác nhận
  thực tế trên test_04, test_09, test_14, test_21 — đúng 100% nhưng bị chấm sai).
- Bỏ khối "has_key_info" cũ (chỉ so từ đơn lẻ, gần như luôn True vì trùng từ
  nối câu tiếng Việt như "là", "của" — không có giá trị phân biệt thật).

Cách dùng:
  python script/pilot_tuan6/semi_auto_grade.py
  → tạo results/tuan6_pilot/grading_filled_auto.csv
  → user kiểm tra lại (dòng "uncertain", và nên liếc qua cả "wrong"/"partial")
  → chạy apply_grading_to_results.py để ghi vào 3 file CSV
"""
import csv
import json
import re
from pathlib import Path
from difflib import SequenceMatcher


def strip_markdown(text: str) -> str:
    """Loại bỏ định dạng markdown và các tiền tố mở đầu thường gặp của model,
    chỉ giữ lại nội dung câu chữ thuần để so sánh công bằng với answer_reference
    (vốn luôn là văn xuôi thuần, không markdown)."""
    t = text
    # Bỏ tiền tố mở đầu kiểu "Dựa trên tài liệu (Điều X):", "Theo tài liệu:"...
    t = re.sub(r'^(Dựa trên|Theo)[^:\n]{0,60}[:：]\s*', '', t.strip(), flags=re.IGNORECASE)
    # Bỏ in đậm/in nghiêng markdown: **text**, *text*
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'\*(.+?)\*', r'\1', t)
    # Bỏ dấu đầu dòng gạch đầu dòng: "- ", "* ", "1. "
    t = re.sub(r'^\s*[-*•]\s+', ' ', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*\d+\.\s+', ' ', t, flags=re.MULTILINE)
    # Bỏ tiêu đề markdown: #, ##...
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)
    # Gộp nhiều khoảng trắng/xuống dòng thành 1 khoảng trắng
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def similarity_ratio(a, b):
    """Tính độ tương đồng giữa 2 string (0-1), sau khi đã làm sạch markdown."""
    return SequenceMatcher(None, strip_markdown(a).lower(), strip_markdown(b).lower()).ratio()


def key_facts_overlap(ref: str, text: str) -> float:
    """Đo riêng tỉ lệ các con số/năm/ngày tháng trong answer_reference có xuất
    hiện lại trong answer_text hay không — với câu hỏi tra cứu số liệu (rất
    phổ biến trong bộ câu hỏi này), số liệu đúng là tín hiệu quan trọng hơn
    độ giống câu chữ tổng thể."""
    ref_numbers = set(re.findall(r'\d[\d.,]*', ref))
    if not ref_numbers:
        return 1.0  # không có số liệu để so, không phạt điểm này
    text_clean = strip_markdown(text)
    matched = sum(1 for n in ref_numbers if n in text_clean)
    return matched / len(ref_numbers)


test_qs = json.loads(Path('data/eval_sets/test_questions.json').read_text(encoding='utf-8'))['questions']
test_by_id = {q['id']: q for q in test_qs}

template_path = Path('results/tuan6_pilot/grading_template.csv')
with open(template_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row.get('is_abstained') == 'True':
        row['answer_correctness_manual'] = 'abstain'
        row['grader_note'] = 'hệ thống từ chối'
        continue

    answer_ref = row.get('answer_reference', '').strip()
    answer_text = row.get('answer_text', '').strip()

    if not answer_ref or not answer_text:
        row['answer_correctness_manual'] = 'skip'
        row['grader_note'] = 'không có dữ liệu để chấm'
        continue

    sim = similarity_ratio(answer_ref, answer_text)
    fact_overlap = key_facts_overlap(answer_ref, answer_text)

    # Kết hợp 2 tín hiệu: độ giống câu chữ (sau khi đã bỏ markdown) VÀ tỉ lệ
    # số liệu khớp đúng — câu trả lời đúng số liệu nhưng diễn đạt khác vẫn
    # nên được coi là "full", không bị phạt vì hình thức.
    if sim > 0.55 or fact_overlap >= 0.9:
        row['answer_correctness_manual'] = 'full'
        row['grader_note'] = f'similarity={sim:.2f}, fact_overlap={fact_overlap:.2f} (cao)'
    elif sim > 0.35 or fact_overlap >= 0.5:
        row['answer_correctness_manual'] = 'partial'
        row['grader_note'] = f'similarity={sim:.2f}, fact_overlap={fact_overlap:.2f}, cần verify chi tiết'
    else:
        row['answer_correctness_manual'] = 'uncertain'
        row['grader_note'] = f'similarity={sim:.2f}, fact_overlap={fact_overlap:.2f}, cần đọc tay'

out_path = Path('results/tuan6_pilot/grading_filled_auto.csv')
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

stats = {}
for row in rows:
    label = row.get('answer_correctness_manual', 'unknown')
    stats[label] = stats.get(label, 0) + 1

print(f"✅ Semi-auto grading hoàn tất (bản đã sửa lỗi markdown)")
print(f"   - File: {out_path}")
print()
print("Phân bổ:")
for label, count in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {label:12}: {count:3} dòng")
print()
print("⚠️  LƯU Ý QUAN TRỌNG: đây vẫn chỉ là GỢI Ý SƠ BỘ, không phải kết quả")
print("   cuối cùng. Ngưỡng 0.55/0.35 chưa được kiểm định kỹ — bạn vẫn PHẢI")
print("   tự đọc lại toàn bộ, đặc biệt các dòng 'partial' và 'uncertain'.")
print()
print("📝 Bước tiếp theo:")
print("  1. Mở file grading_filled_auto.csv")
print("  2. Đọc lại TỪNG dòng, so answer_text với answer_reference bằng mắt")
print("  3. Sửa lại answer_correctness_manual nếu cần (full/partial/wrong)")
print("  4. Lưu file")
print("  5. Chạy: python script/pilot_tuan6/apply_grading_to_results.py --template results/tuan6_pilot/grading_filled_auto.csv")
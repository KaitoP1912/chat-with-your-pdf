#!/usr/bin/env python3
"""
Tạo template chấm tay answer_correctness cho dữ liệu Tuần 8 (50 câu × 3 cấu hình).

Cách sử dụng:
  1. Chạy script này để tạo file 'grading_template_50cau.csv'
  2. Mở file trong Excel, điền cột 'answer_correctness_manual' bằng: full / partial / wrong
  3. Lưu file
  4. Chạy script 'semi_auto_grade_tuan8.py' để gợi ý chấm tự động
"""
import csv
import json
from pathlib import Path

test_qs = json.loads(Path('data/eval_sets/test_questions.json').read_text(encoding='utf-8'))['questions']
test_by_id = {q['id']: q for q in test_qs}

template_rows = []
for config in ['page_aware', 'fixed_size', 'longcontext']:
    csv_file = Path(f'results/tuan8/test_qa_results_{config}_50cau.csv')
    with open(csv_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        q_id = row['id']
        q_data = test_by_id.get(q_id, {})

        if not row.get('answer_text', '').strip():
            continue

        template_rows.append({
            'id': q_id,
            'config': config,
            'question': q_data.get('question', ''),
            'answer_reference': q_data.get('answer_reference', ''),
            'answer_text': row.get('answer_text', ''),
            'is_abstained': row.get('is_abstained', ''),
            'answer_correctness_manual': '',
            'grader_note': '',
        })

out_path = Path('results/tuan8/grading_template_50cau.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    'id', 'config', 'question', 'answer_reference', 'answer_text',
    'is_abstained', 'answer_correctness_manual', 'grader_note'
]

with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(template_rows)

print(f"✅ Tạo template chấm tay Tuần 8")
print(f"   - File: {out_path}")
print(f"   - Tổng dòng: {len(template_rows)}")
print()
print("📝 Hướng dẫn:")
print("  1. Mở file grading_template_50cau.csv trong Excel")
print("  2. Điền cột 'answer_correctness_manual' bằng 1 trong 3 giá trị:")
print("     - 'full':    đúng hoàn toàn")
print("     - 'partial': đúng một phần")
print("     - 'wrong':   sai")
print("  3. (Tùy chọn) Điền cột 'grader_note' để ghi chú ngắn")
print("  4. Lưu file")
print("  5. Chạy: python script/pilot_tuan8/semi_auto_grade_tuan8.py")
